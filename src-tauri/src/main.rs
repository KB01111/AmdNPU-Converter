#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    npu_model_converter_lib::run();
}
