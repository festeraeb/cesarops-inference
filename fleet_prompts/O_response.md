```diff
--- src/server.rs
+++ src/server.rs
@@ -1,10 +1,12 @@
-use axum::{extract::State, response::Json, routing::{get, post}, Router};
+use axum::{extract::State, response::Json, routing::{get, post}, Router, Json};
 use serde::{Deserialize, Serialize};
 use std::sync::Arc;
 use tokio::sync::Mutex;
+use parking_lot::RwLock;
+use crate::kv_prefix_cache::{KvPrefixCache, hash_sequence, KvSlice};
 
 pub struct InferenceState {
     pub model: Arc<Model>,
     pub tokenizer: Arc<Tokenizer>,
+    pub prefix_cache: Arc<RwLock<KvPrefixCache>>,
 }
 
 #[derive(Serialize)]
@@ -18,6 +20,14 @@
     pub model_name: String,
     pub cache_stats: Option<CacheStats>,
 }
+
+#[derive(Serialize, Clone)]
+pub struct CacheStats {
+    pub hit_rate: f64,
+    pub capacity: usize,
+    pub total_tokens: u64,
+}
 
 async fn get_model_info(State(state): State<Arc<InferenceState>>) -> Json<ModelInfo> {
-    Json(ModelInfo { ... })
+    let stats = state.prefix_cache.read().stats();
+    Json(ModelInfo {
+        model_name: state.model.name.clone(),
+        cache_stats: Some(CacheStats {
+            hit_rate: stats.hit_rate,
+            capacity: state.prefix_cache.read().capacity(),
+            total_tokens: stats.total_tokens,
+        }),
+    })
 }
 
 async fn generate(State(state): State<Arc<InferenceState>>, Json(request): Json<GenerateRequest>) -> Json<GenerateResponse> {
     let prompt_tokens = tokenizer.encode(&request.prompt);
+    let prompt_hashes = hash_sequence(&prompt_tokens, 0);
+    
+    // V1: Telemetry and Bookkeeping
+    {
+        let mut cache = state.prefix_cache.write();
+        if let Some((len, _slice)) = cache.prefix_match(&prompt_hashes) {
+            tracing::info!("[cache] hit prefix_len={}", len);
+        } else {
+            tracing::info!("[cache] miss");
+        }
+    }
+
     let mut kv_cache = KvCache::new(num_layers, max_seq_len);
     let mut hidden_state = embed(prompt_tokens[0]);
 
@@ -35,5 +45,13 @@
     // Generation loop:
     for step in 0..max_tokens { ... }
 
+    // Commit the prefix to the cache for future requests
+    {
+        let mut cache = state.prefix_cache.write();
+        cache.commit(&prompt_hashes, KvSlice { 
+            start_pos: 0, 
+            len: prompt_tokens.len() as u32, 
+            layer_data_handle: 0 
+        });
+    }
+
     // Decode + return
     Json(GenerateResponse { ... })
 }
 
-#[tokio::main]
-async fn main() {
-    // ... setup code ...
-    let state = Arc::new(InferenceState {
-        model,
-        tokenizer,
-        // ...
-    });
+async fn run_server() {
+    // ... setup code ...
+    let prefix_cache = Arc::new(RwLock::new(KvPrefixCache::from_vram_budget(
+        16 * 1024 * 1024 * 1024, 
+        57344
+    )));
+
+    let state = Arc::new(InferenceState {
+        model,
+        tokenizer,
+        prefix_cache,
+    });
+
+    let app = Router::new()
+        .route("/api/v1/model", get(get_model_info))
+        .route("/api/v1/generate", post(generate))
+        .with_state(state);
+    // ... axum serve ...
 }
```
