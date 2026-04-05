use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Qwen,
    Gemma4,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantOverride {
    pub bits: u32,
    pub group_size: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantizationConfig {
    pub bits: u32,
    pub group_size: u32,
    pub mode: Option<String>,
    /// Per-component overrides keyed by weight path (e.g. "language_model.model.layers.0.self_attn.q_proj")
    pub overrides: HashMap<String, QuantOverride>,
}

impl QuantizationConfig {
    /// Look up bits for a specific weight path, falling back to the global default.
    pub fn bits_for(&self, path: &str) -> i32 {
        self.overrides
            .get(path)
            .map(|o| o.bits as i32)
            .unwrap_or(self.bits as i32)
    }

    /// Look up group_size for a specific weight path.
    pub fn group_size_for(&self, path: &str) -> i32 {
        self.overrides
            .get(path)
            .map(|o| o.group_size as i32)
            .unwrap_or(self.group_size as i32)
    }
}

impl<'de> Deserialize<'de> for QuantizationConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::Map::deserialize(deserializer)?;

        let bits = map
            .get("bits")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as u32;
        let group_size = map
            .get("group_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as u32;
        let mode = map
            .get("mode")
            .and_then(|v| v.as_str())
            .map(String::from);

        let mut overrides = HashMap::new();
        for (key, value) in &map {
            if matches!(key.as_str(), "bits" | "group_size" | "mode") {
                continue;
            }
            if let Ok(ovr) = serde_json::from_value::<QuantOverride>(value.clone()) {
                overrides.insert(key.clone(), ovr);
            }
        }

        Ok(QuantizationConfig {
            bits,
            group_size,
            mode,
            overrides,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeTypeConfig {
    #[serde(default = "default_rope_theta_10k")]
    pub rope_theta: f64,
    #[serde(default = "default_1_0")]
    pub partial_rotary_factor: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub full_attention: Option<RopeTypeConfig>,
    #[serde(default)]
    pub sliding_attention: Option<RopeTypeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TextModelArgs {
    #[serde(default)]
    pub model_type: Option<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    pub num_experts: usize,
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,

    // MoE top-k: Qwen uses num_experts_per_tok, Gemma4 uses top_k_experts
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,

    // Linear attention (GatedDeltaNet) — Qwen only
    #[serde(default = "default_32")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_16")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_128")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_128")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_4")]
    pub linear_conv_kernel_dim: usize,

    // RoPE
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    // Gemma4-specific
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub enable_moe_block: bool,

    // EOS tokens
    #[serde(default)]
    pub eos_token_id: EosTokenId,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl Default for EosTokenId {
    fn default() -> Self {
        EosTokenId::Multiple(vec![248046, 248044])
    }
}

impl EosTokenId {
    #[allow(dead_code)]
    pub fn ids(&self) -> Vec<u32> {
        match self {
            EosTokenId::Single(id) => vec![*id],
            EosTokenId::Multiple(ids) => ids.clone(),
        }
    }
}

fn default_rope_theta() -> f64 {
    10_000_000.0
}
fn default_rope_theta_10k() -> f64 {
    10_000.0
}
fn default_1_0() -> f64 {
    1.0
}
fn default_partial_rotary() -> f64 {
    0.25
}
fn default_true() -> bool {
    true
}
fn default_full_attn_interval() -> usize {
    4
}
fn default_32() -> usize {
    32
}
fn default_16() -> usize {
    16
}
fn default_128() -> usize {
    128
}
fn default_4() -> usize {
    4
}

impl TextModelArgs {
    pub fn from_config_file(path: &Path) -> anyhow::Result<(Self, Option<QuantizationConfig>)> {
        let content = std::fs::read_to_string(path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        let text_config = config.get("text_config").unwrap_or(&config);
        let args: TextModelArgs = serde_json::from_value(text_config.clone())?;

        let quant = config
            .get("quantization")
            .map(|q| serde_json::from_value(q.clone()))
            .transpose()?;

        Ok((args, quant))
    }

    pub fn model_type(&self) -> ModelType {
        match self.model_type.as_deref() {
            Some(t) if t.starts_with("gemma4") => ModelType::Gemma4,
            _ => ModelType::Qwen,
        }
    }

    pub fn experts_per_tok(&self) -> usize {
        self.top_k_experts
            .or(self.num_experts_per_tok)
            .unwrap_or(8)
    }

    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        match &self.layer_types {
            Some(types) => types[layer_idx] == "full_attention",
            None => !self.is_linear_layer(layer_idx),
        }
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f64 * self.partial_rotary_factor) as i32
    }

    /// Get RoPE config for a Gemma4 layer (rope_dims, rope_theta).
    pub fn gemma4_rope_config(&self, is_full: bool) -> (i32, f32) {
        let params = self.rope_parameters.as_ref();
        let cfg = if is_full {
            params.and_then(|p| p.full_attention.as_ref())
        } else {
            params.and_then(|p| p.sliding_attention.as_ref())
        };
        let theta = cfg.map(|c| c.rope_theta).unwrap_or(10000.0) as f32;
        let partial = cfg.map(|c| c.partial_rotary_factor).unwrap_or(1.0);
        let head_dim = if is_full {
            self.global_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        };
        let dims = (head_dim as f64 * partial) as i32;
        (dims, theta)
    }

    /// Key dimension for linear attention: num_k_heads * key_head_dim
    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Value dimension for linear attention: num_v_heads * value_head_dim
    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Conv dimension for linear attention: key_dim*2 + value_dim
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}
