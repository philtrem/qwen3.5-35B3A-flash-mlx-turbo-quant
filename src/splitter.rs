use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

/// Expert layout detected from tensor naming conventions.
enum ExpertLayout {
    /// Qwen: language_model.model.layers.{i}.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
    /// 9 tensors per layer, each (num_experts, d1, d2).
    Qwen,
    /// Gemma 4 (mlx-community): model.language_model.layers.{i}.experts.{gate_up_proj,down_proj}[_{scales,biases}]
    /// 6 tensors per layer; gate_up_proj is fused (first half = gate, second half = up).
    /// Unfused to 9 tensors during ECB writing for compatibility.
    Gemma4,
    /// Gemma 4 (Unsloth UD): language_model.model.layers.{i}.experts.switch_glu.{gate,up,down}_proj.{weight,scales,biases}
    /// 9 tensors per layer, already unfused.
    Gemma4Ud,
}

/// Detect expert layout from weight map keys.
fn detect_layout(weight_map: &serde_json::Map<String, Value>) -> ExpertLayout {
    for key in weight_map.keys() {
        if key.contains("switch_mlp") {
            return ExpertLayout::Qwen;
        }
        if key.contains(".experts.switch_glu.") {
            return ExpertLayout::Gemma4Ud;
        }
        if key.contains(".experts.gate_up_proj") || key.contains(".experts.down_proj") {
            return ExpertLayout::Gemma4;
        }
    }
    // Default to Qwen for backward compat
    ExpertLayout::Qwen
}

/// Check if a tensor key is an expert tensor.
fn is_expert_tensor(key: &str, layout: &ExpertLayout) -> bool {
    match layout {
        ExpertLayout::Qwen => key.contains("switch_mlp"),
        ExpertLayout::Gemma4 => {
            // Match experts.gate_up_proj, experts.down_proj, and their _scales/_biases variants
            key.contains(".experts.gate_up_proj") || key.contains(".experts.down_proj")
        }
        ExpertLayout::Gemma4Ud => key.contains(".experts.switch_glu."),
    }
}

/// Discover the number of layers that have expert tensors.
fn discover_num_expert_layers(weight_map: &serde_json::Map<String, Value>, layout: &ExpertLayout) -> u32 {
    let layer_prefix = match layout {
        ExpertLayout::Qwen | ExpertLayout::Gemma4Ud => "language_model.model.layers.",
        ExpertLayout::Gemma4 => "model.language_model.layers.",
    };
    let expert_marker = match layout {
        ExpertLayout::Qwen => "switch_mlp",
        ExpertLayout::Gemma4 | ExpertLayout::Gemma4Ud => ".experts.",
    };

    let mut max_layer = 0u32;
    for key in weight_map.keys() {
        if key.contains(expert_marker) {
            if let Some(rest) = key.strip_prefix(layer_prefix) {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<u32>() {
                        max_layer = max_layer.max(idx);
                    }
                }
            }
        }
    }
    max_layer + 1
}

/// Canonicalize a resident tensor name: strip model-specific prefix to produce
/// names starting with "language_model.layers.N..." for all layouts.
fn canonicalize_resident_name(name: &str, layout: &ExpertLayout) -> String {
    match layout {
        ExpertLayout::Qwen => name
            .strip_prefix("language_model.")
            .unwrap_or(name)
            .to_string(),
        ExpertLayout::Gemma4 => name
            .strip_prefix("model.")
            .unwrap_or(name)
            .to_string(),
        // language_model.model.layers.N.X → language_model.layers.N.X
        // language_model.model.embed_tokens.X → language_model.embed_tokens.X
        ExpertLayout::Gemma4Ud => name
            .replacen("language_model.model.", "language_model.", 1),
    }
}

/// Split a model into resident weights (safetensors) and per-layer expert files.
/// `format` is "ecb" (expert-centric binary) or "safetensors".
pub fn split_model(model_path: &Path, output_path: &Path, format: &str) -> io::Result<()> {
    fs::create_dir_all(output_path)?;
    let resident_dir = output_path.join("resident");
    let expert_dir = output_path.join("experts");
    fs::create_dir_all(&resident_dir)?;
    fs::create_dir_all(&expert_dir)?;

    // Copy config files
    for name in &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        let src = model_path.join(name);
        if src.exists() {
            fs::copy(&src, output_path.join(name))?;
        }
    }

    // Read weight index
    let index_path = model_path.join("model.safetensors.index.json");
    let index_str = fs::read_to_string(&index_path)?;
    let index: Value = serde_json::from_str(&index_str)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no weight_map in index"))?;

    let layout = detect_layout(weight_map);
    let layout_name = match &layout {
        ExpertLayout::Qwen => "Qwen (switch_mlp, separate gate/up)",
        ExpertLayout::Gemma4 => "Gemma 4 (experts, fused gate_up_proj)",
        ExpertLayout::Gemma4Ud => "Gemma 4 UD (switch_glu, separate gate/up)",
    };
    eprintln!("Detected expert layout: {}", layout_name);

    // Classify tensors
    let mut resident_tensors: Vec<String> = Vec::new();
    let mut expert_tensors: Vec<String> = Vec::new();
    for key in weight_map.keys() {
        if is_expert_tensor(key, &layout) {
            expert_tensors.push(key.clone());
        } else {
            resident_tensors.push(key.clone());
        }
    }

    eprintln!(
        "Found {} resident tensors, {} expert tensors",
        resident_tensors.len(),
        expert_tensors.len()
    );

    // Open all shard files
    let mut shard_mmaps: HashMap<String, Mmap> = HashMap::new();
    let mut shard_files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    for shard_name in &shard_files {
        let shard_path = model_path.join(shard_name);
        let file = File::open(&shard_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        shard_mmaps.insert(shard_name.clone(), mmap);
    }

    // Step 1: Write resident weights as safetensors
    write_resident_weights(&shard_mmaps, weight_map, &resident_tensors, &resident_dir, &layout)?;

    // Step 2: Write expert weights in requested format
    match format {
        "ecb" => write_expert_ecb(&shard_mmaps, weight_map, &expert_dir, &layout)?,
        "safetensors" => return Err(io::Error::new(io::ErrorKind::InvalidInput,
            "safetensors expert format is no longer supported (inference requires ECB for zero-copy Metal buffers). Use --format ecb (the default).")),
        _ => return Err(io::Error::new(io::ErrorKind::InvalidInput,
            format!("unknown format '{}', expected 'ecb'", format))),
    }

    // Write split metadata
    let meta = serde_json::json!({
        "original_model": model_path.to_str(),
        "resident_dir": "resident",
        "expert_dir": "experts",
        "format": format,
    });
    fs::write(
        output_path.join("split_config.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )?;

    eprintln!("Model split complete: {}", output_path.display());
    Ok(())
}

/// Parse a safetensors shard and return (header_json, data_offset).
fn parse_shard(mmap: &[u8]) -> io::Result<(Value, usize)> {
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&mmap[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok((header, 8 + header_size))
}

/// Extract raw tensor bytes from a shard mmap.
fn extract_tensor<'a>(mmap: &'a [u8], header: &Value, tensor_name: &str) -> io::Result<&'a [u8]> {
    let info = header.get(tensor_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("tensor {} not in shard", tensor_name),
        )
    })?;
    let offsets = info["data_offsets"]
        .as_array()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no data_offsets"))?;
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;

    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_base = 8 + header_size;

    Ok(&mmap[data_base + start..data_base + end])
}

/// Get tensor shape from shard header.
fn tensor_shape(header: &Value, tensor_name: &str) -> io::Result<Vec<usize>> {
    let info = header
        .get(tensor_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, tensor_name.to_string()))?;
    Ok(info["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect())
}

/// Get tensor dtype from shard header.
fn tensor_dtype(header: &Value, tensor_name: &str) -> io::Result<String> {
    let info = header
        .get(tensor_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, tensor_name.to_string()))?;
    Ok(info["dtype"].as_str().unwrap_or("F32").to_string())
}

fn parse_all_shard_headers(shard_mmaps: &HashMap<String, Mmap>) -> io::Result<HashMap<String, Value>> {
    let mut headers = HashMap::new();
    for (name, mmap) in shard_mmaps {
        let (header, _) = parse_shard(mmap)?;
        headers.insert(name.clone(), header);
    }
    Ok(headers)
}

/// Write resident (non-expert) weights as safetensors.
fn write_resident_weights(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    resident_tensors: &[String],
    output_dir: &Path,
    layout: &ExpertLayout,
) -> io::Result<()> {
    let shard_headers = parse_all_shard_headers(shard_mmaps)?;

    let mut tensor_data: Vec<(String, Vec<u8>, String, Vec<usize>)> = Vec::new();

    for tensor_name in resident_tensors {
        let shard_name = weight_map[tensor_name].as_str().unwrap();
        let mmap = &shard_mmaps[shard_name];
        let header = &shard_headers[shard_name];

        let data = extract_tensor(mmap, header, tensor_name)?;
        let shape = tensor_shape(header, tensor_name)?;
        let dtype = tensor_dtype(header, tensor_name)?;

        let clean_name = canonicalize_resident_name(tensor_name, layout);

        tensor_data.push((clean_name, data.to_vec(), dtype, shape));
    }

    tensor_data.sort_by(|a, b| a.0.cmp(&b.0));
    write_safetensors_file(&tensor_data, &output_dir.join("resident.safetensors"))?;

    // Write index file
    let mut new_weight_map = serde_json::Map::new();
    for (name, _, _, _) in &tensor_data {
        new_weight_map.insert(name.clone(), Value::String("resident.safetensors".to_string()));
    }
    let index = serde_json::json!({
        "metadata": { "format": "mlx" },
        "weight_map": new_weight_map,
    });
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )?;

    let total_bytes: usize = tensor_data.iter().map(|(_, d, _, _)| d.len()).sum();
    eprintln!(
        "Wrote {} resident tensors ({:.2} GB)",
        tensor_data.len(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    Ok(())
}

/// ECB dtype encoding (matches memory.rs parser).
fn ecb_dtype_code(dtype_str: &str) -> u32 {
    match dtype_str {
        "U8" => 0,
        "U32" => 1,
        "BF16" => 2,
        "F16" => 3,
        "F32" => 4,
        "I32" => 5,
        _ => panic!("unsupported dtype for ECB: {}", dtype_str),
    }
}

/// Per-tensor metadata for ECB writing.
struct TensorMeta {
    data: Vec<u8>,
    per_expert_stride: usize,
    dtype_code: u32,
    expert_shape: Vec<u32>,
    num_experts: usize,
}

/// Load a tensor from shards and build its TensorMeta.
fn load_tensor_meta(
    tensor_name: &str,
    weight_map: &serde_json::Map<String, Value>,
    shard_mmaps: &HashMap<String, Mmap>,
    shard_headers: &HashMap<String, Value>,
) -> io::Result<TensorMeta> {
    let shard_name = weight_map.get(tensor_name).and_then(|v| v.as_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound,
            format!("tensor {} not in weight map", tensor_name)))?;
    let mmap = &shard_mmaps[shard_name];
    let header = &shard_headers[shard_name];

    let data = extract_tensor(mmap, header, tensor_name)?;
    let shape = tensor_shape(header, tensor_name)?;
    let dtype_str = tensor_dtype(header, tensor_name)?;

    let num_experts = shape[0];
    let expert_shape: Vec<u32> = shape[1..].iter().map(|&s| s as u32).collect();
    let per_expert_stride = data.len() / num_experts;

    Ok(TensorMeta {
        data: data.to_vec(),
        per_expert_stride,
        dtype_code: ecb_dtype_code(&dtype_str),
        expert_shape,
        num_experts,
    })
}

/// Split a fused (gate_up) TensorMeta in half along dim 0 (first dim after expert dim).
/// Returns (gate_half, up_half).
fn split_fused_gate_up(fused: TensorMeta) -> (TensorMeta, TensorMeta) {
    let num_experts = fused.num_experts;
    let fused_dim0 = fused.expert_shape[0] as usize;
    assert!(fused_dim0 % 2 == 0, "fused gate_up dim must be even, got {}", fused_dim0);
    let half_dim0 = fused_dim0 / 2;

    // Per-expert byte layout: fused_dim0 rows, each row = stride_per_row bytes
    let stride_per_row = fused.per_expert_stride / fused_dim0;
    let half_stride = half_dim0 * stride_per_row;

    let mut gate_data = Vec::with_capacity(num_experts * half_stride);
    let mut up_data = Vec::with_capacity(num_experts * half_stride);

    for e in 0..num_experts {
        let expert_start = e * fused.per_expert_stride;
        let gate_start = expert_start;
        let up_start = expert_start + half_stride;
        gate_data.extend_from_slice(&fused.data[gate_start..gate_start + half_stride]);
        up_data.extend_from_slice(&fused.data[up_start..up_start + half_stride]);
    }

    let mut half_shape = fused.expert_shape.clone();
    half_shape[0] = half_dim0 as u32;

    let gate = TensorMeta {
        data: gate_data,
        per_expert_stride: half_stride,
        dtype_code: fused.dtype_code,
        expert_shape: half_shape.clone(),
        num_experts,
    };
    let up = TensorMeta {
        data: up_data,
        per_expert_stride: half_stride,
        dtype_code: fused.dtype_code,
        expert_shape: half_shape,
        num_experts,
    };

    (gate, up)
}

/// Write ECB header and expert-centric data from a list of TensorMeta.
fn write_ecb_file(tensor_metas: &[TensorMeta], file_path: &Path) -> io::Result<()> {
    let num_experts = tensor_metas[0].num_experts as u32;
    let per_expert_stride: usize = tensor_metas.iter().map(|t| t.per_expert_stride).sum();
    let num_tensors = tensor_metas.len() as u32;
    let header_size: u32 = 16384;

    let mut header_buf = vec![0u8; header_size as usize];
    let mut pos = 0usize;

    // Magic
    header_buf[pos..pos + 4].copy_from_slice(b"ECB1");
    pos += 4;
    // num_experts
    header_buf[pos..pos + 4].copy_from_slice(&num_experts.to_le_bytes());
    pos += 4;
    // per_expert_stride
    header_buf[pos..pos + 4].copy_from_slice(&(per_expert_stride as u32).to_le_bytes());
    pos += 4;
    // num_tensors
    header_buf[pos..pos + 4].copy_from_slice(&num_tensors.to_le_bytes());
    pos += 4;
    // header_size
    header_buf[pos..pos + 4].copy_from_slice(&header_size.to_le_bytes());
    pos += 4;

    // Tensor descriptors
    for t in tensor_metas {
        header_buf[pos..pos + 4].copy_from_slice(&(t.per_expert_stride as u32).to_le_bytes());
        pos += 4;
        header_buf[pos..pos + 4].copy_from_slice(&t.dtype_code.to_le_bytes());
        pos += 4;
        header_buf[pos..pos + 4].copy_from_slice(&(t.expert_shape.len() as u32).to_le_bytes());
        pos += 4;
        for &dim in &t.expert_shape {
            header_buf[pos..pos + 4].copy_from_slice(&dim.to_le_bytes());
            pos += 4;
        }
    }

    let mut file = File::create(file_path)?;
    file.write_all(&header_buf)?;

    // Expert-centric data
    for expert_idx in 0..num_experts as usize {
        for t in tensor_metas {
            let start = expert_idx * t.per_expert_stride;
            let end = start + t.per_expert_stride;
            file.write_all(&t.data[start..end])?;
        }
    }

    file.sync_all()?;
    Ok(())
}

/// Write per-layer expert-centric binary (ECB) files.
///
/// Rearranges tensor-centric layout into expert-centric layout so that
/// all data for one expert is contiguous (single pread per expert).
///
/// Output always has 9 tensors per layer in gate/up/down × weight/scales/biases
/// order. For Gemma 4's fused gate_up_proj, the tensor is split in half.
fn write_expert_ecb(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    expert_dir: &Path,
    layout: &ExpertLayout,
) -> io::Result<()> {
    let shard_headers = parse_all_shard_headers(shard_mmaps)?;
    let num_layers = discover_num_expert_layers(weight_map, layout);

    eprintln!("Writing {} layers of expert ECB files...", num_layers);

    for layer_idx in 0..num_layers {
        let tensor_metas = match layout {
            ExpertLayout::Qwen => collect_qwen_tensors(layer_idx, weight_map, shard_mmaps, &shard_headers)?,
            ExpertLayout::Gemma4 => collect_gemma4_tensors(layer_idx, weight_map, shard_mmaps, &shard_headers)?,
            ExpertLayout::Gemma4Ud => collect_gemma4_ud_tensors(layer_idx, weight_map, shard_mmaps, &shard_headers)?,
        };

        let num_experts = tensor_metas[0].num_experts;
        let per_expert_stride: usize = tensor_metas.iter().map(|t| t.per_expert_stride).sum();

        let file_path = expert_dir.join(format!("layer_{:02}_experts.ecb", layer_idx));
        write_ecb_file(&tensor_metas, &file_path)?;

        if (layer_idx + 1) % 10 == 0 || layer_idx == num_layers - 1 {
            eprintln!("  Wrote layer {}/{} ({:.1} MB, {} experts)",
                layer_idx + 1, num_layers,
                (16384 + num_experts * per_expert_stride) as f64 / 1e6,
                num_experts);
        }
    }

    Ok(())
}

/// Collect 9 TensorMetas for a Qwen layer (gate/up/down × weight/scales/biases).
fn collect_qwen_tensors(
    layer_idx: u32,
    weight_map: &serde_json::Map<String, Value>,
    shard_mmaps: &HashMap<String, Mmap>,
    shard_headers: &HashMap<String, Value>,
) -> io::Result<Vec<TensorMeta>> {
    let prefix = format!("language_model.model.layers.{}.mlp.switch_mlp", layer_idx);
    let proj_names = ["gate_proj", "up_proj", "down_proj"];
    let comp_names = ["weight", "scales", "biases"];

    let mut metas = Vec::with_capacity(9);
    for proj in &proj_names {
        for comp in &comp_names {
            let name = format!("{}.{}.{}", prefix, proj, comp);
            metas.push(load_tensor_meta(&name, weight_map, shard_mmaps, shard_headers)?);
        }
    }
    Ok(metas)
}

/// Collect 9 TensorMetas for a Gemma 4 layer.
/// Unfuses gate_up_proj into separate gate and up tensors.
fn collect_gemma4_tensors(
    layer_idx: u32,
    weight_map: &serde_json::Map<String, Value>,
    shard_mmaps: &HashMap<String, Mmap>,
    shard_headers: &HashMap<String, Value>,
) -> io::Result<Vec<TensorMeta>> {
    let prefix = format!("model.language_model.layers.{}.experts", layer_idx);
    let comp_suffixes = ["", "_scales", "_biases"]; // "" = the quantized weight itself

    let mut metas = Vec::with_capacity(9);

    // gate_up_proj → split into gate_proj + up_proj
    for suffix in &comp_suffixes {
        let name = format!("{}.gate_up_proj{}", prefix, suffix);
        let fused = load_tensor_meta(&name, weight_map, shard_mmaps, shard_headers)?;
        let (gate, up) = split_fused_gate_up(fused);
        // Defer: push gate now, stash up for later
        metas.push(gate);
        metas.push(up);
    }

    // At this point metas has: gate.w, up.w, gate.s, up.s, gate.b, up.b
    // But we need: gate.w, gate.s, gate.b, up.w, up.s, up.b, down.w, down.s, down.b
    // Reorder: indices 0,2,4 (gate w/s/b), 1,3,5 (up w/s/b)
    let reordered_gate_up: Vec<TensorMeta> = {
        let mut v = Vec::with_capacity(6);
        // Move all out first so we can index freely
        let mut all: Vec<TensorMeta> = metas.drain(..).collect();
        // gate: w=0, s=2, b=4
        v.push(all.swap_remove(0)); // idx 0 → gate.w (now: [up.w, gate.s, up.s, gate.b, up.b])
        // After swap_remove(0): up.b moved to 0. Order: [up.b, gate.s, up.s, gate.b, up.w_gone]
        // This is getting messy. Let me just collect into a vec and index.
        drop(v);
        drop(all);
        Vec::new()
    };
    // Cleaner approach: collect all 6, then reorder
    let _ = reordered_gate_up;
    metas.clear();

    let mut gate_w_s_b = Vec::with_capacity(3);
    let mut up_w_s_b = Vec::with_capacity(3);
    for suffix in &comp_suffixes {
        let name = format!("{}.gate_up_proj{}", prefix, suffix);
        let fused = load_tensor_meta(&name, weight_map, shard_mmaps, shard_headers)?;
        let (gate, up) = split_fused_gate_up(fused);
        gate_w_s_b.push(gate);
        up_w_s_b.push(up);
    }

    // down_proj
    let mut down_w_s_b = Vec::with_capacity(3);
    for suffix in &comp_suffixes {
        let name = format!("{}.down_proj{}", prefix, suffix);
        down_w_s_b.push(load_tensor_meta(&name, weight_map, shard_mmaps, shard_headers)?);
    }

    // Final order: gate.w, gate.s, gate.b, up.w, up.s, up.b, down.w, down.s, down.b
    metas.extend(gate_w_s_b);
    metas.extend(up_w_s_b);
    metas.extend(down_w_s_b);

    Ok(metas)
}

/// Collect 9 TensorMetas for a Gemma 4 UD layer (already unfused switch_glu).
/// Order: gate.w, gate.s, gate.b, up.w, up.s, up.b, down.w, down.s, down.b
fn collect_gemma4_ud_tensors(
    layer_idx: u32,
    weight_map: &serde_json::Map<String, Value>,
    shard_mmaps: &HashMap<String, Mmap>,
    shard_headers: &HashMap<String, Value>,
) -> io::Result<Vec<TensorMeta>> {
    let prefix = format!("language_model.model.layers.{}.experts.switch_glu", layer_idx);
    let proj_names = ["gate_proj", "up_proj", "down_proj"];
    let comp_names = ["weight", "scales", "biases"];

    let mut metas = Vec::with_capacity(9);
    for proj in &proj_names {
        for comp in &comp_names {
            let name = format!("{}.{}.{}", prefix, proj, comp);
            metas.push(load_tensor_meta(&name, weight_map, shard_mmaps, shard_headers)?);
        }
    }
    Ok(metas)
}

/// Write a safetensors file from tensor data.
fn write_safetensors_file(
    tensors: &[(String, Vec<u8>, String, Vec<usize>)],
    path: &Path,
) -> io::Result<()> {
    let mut header_map = serde_json::Map::new();
    let mut offset = 0u64;

    for (name, data, dtype, shape) in tensors {
        let end = offset + data.len() as u64;
        header_map.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }

    header_map.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt"}),
    );

    let header_json = serde_json::to_string(&Value::Object(header_map))?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;
    for (_, data, _, _) in tensors {
        file.write_all(data)?;
    }
    file.sync_all()?;

    Ok(())
}
