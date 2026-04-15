mod sidecar;

use sidecar::SidecarState;
use std::sync::Mutex;
use tauri::{AppHandle, Manager};
use tauri_plugin_store::StoreExt;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ServerConfig {
    pub mode: String,       // "local" | "remote"
    pub remote_url: String,
    pub hf_token: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            mode: "local".into(),
            remote_url: String::new(),
            hf_token: String::new(),
        }
    }
}

fn load_config(app: &AppHandle) -> ServerConfig {
    let store = app.store("config.json").unwrap();
    store
        .get("server_config")
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

fn save_config(app: &AppHandle, config: &ServerConfig) {
    let store = app.store("config.json").unwrap();
    store.set("server_config", serde_json::to_value(config).unwrap());
    let _ = store.save();
}

#[tauri::command]
fn read_file(path: String) -> Result<Vec<u8>, String> {
    std::fs::read(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_server_config(app: AppHandle) -> ServerConfig {
    load_config(&app)
}

#[tauri::command]
fn set_server_config(app: AppHandle, config: ServerConfig) {
    let old = load_config(&app);
    save_config(&app, &config);

    match config.mode.as_str() {
        "local" if old.mode != "local" => {
            let token = if config.hf_token.is_empty() {
                None
            } else {
                Some(config.hf_token.clone())
            };
            let _ = sidecar::start(&app, token);
        }
        "remote" if old.mode != "remote" => {
            sidecar::stop(&app);
        }
        _ => {}
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(SidecarState(Mutex::new(None)))
        .plugin(tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .setup(|app| {
            let config = load_config(app.handle());
            if config.mode == "local" {
                let token = if config.hf_token.is_empty() {
                    None
                } else {
                    Some(config.hf_token.clone())
                };
                // Best-effort — sidecar binary may not exist yet in dev mode
                if let Err(e) = sidecar::start(app.handle(), token) {
                    log::warn!("Could not start sidecar: {e}");
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                sidecar::stop(window.app_handle());
            }
        })
        .invoke_handler(tauri::generate_handler![read_file, get_server_config, set_server_config])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
