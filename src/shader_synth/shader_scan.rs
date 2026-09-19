use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct ShaderEntry {
    pub name: String,
    pub path: PathBuf,
    pub kind: String,
}

pub fn scan_shaders(dir: &Path) -> Vec<ShaderEntry> {
    let mut out = vec![];
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                out.extend(scan_shaders(&path));
            } else if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy();
                if ext_str == "wgsl" || ext_str == "glsl" || ext_str == "comp" || ext_str == "spv" {
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    let mut kind = detect_shader_kind(&name);
                    // Coopmat kernels get classified by source content too —
                    // a GLSL kernel may not have "coopmat" in the filename
                    // but uses `coopMatMulAddKHR` or `gl_MatrixUseA`.
                    if kind != "coopmat" && (ext_str == "glsl" || ext_str == "comp") {
                        if let Ok(src) = fs::read_to_string(&path) {
                            if src.contains("coopMatMulAdd")
                                || src.contains("CooperativeMatrix")
                                || src.contains("gl_MatrixUseA")
                            {
                                kind = "coopmat".to_string();
                            }
                        }
                    }
                    out.push(ShaderEntry { name, path, kind });
                }
            }
        }
    }
    out
}

fn detect_shader_kind(name: &str) -> String {
    if name.contains("coopmat") { "coopmat".to_string() }
    else if name.contains("iq4") { "iq4_xs".to_string() }
    else if name.contains("q6k") { "q6_k".to_string() }
    else if name.contains("q4") { "q4".to_string() }
    else if name.contains("fp16") || name.contains("half") { "fp16".to_string() }
    else if name.contains("int8") { "int8".to_string() }
    else if name.contains("moe") { "moe".to_string() }
    else if name.contains("attention") || name.contains("attn") { "attention".to_string() }
    else if name.contains("matmul") || name.contains("matvec") { "matmul".to_string() }
    else { "generic".to_string() }
}
