import vitis
from pathlib import Path

xsa_name = "microblaze_kria_design_wrapper.xsa"
script_dir = Path(__file__).resolve().parent.parent
workspace_dir = script_dir / "circuits/microblaze/vitis-workspace"

xsa_path = workspace_dir.parent / xsa_name
source_dir = Path(__file__).resolve().parent / "microblaze_software"

if not xsa_path.is_file():
    raise FileNotFoundError(f"Expected the exported XSA at {xsa_path}")

print(f"Creating Vitis project in {workspace_dir}")

client = vitis.create_client()
client.set_workspace(path=str(workspace_dir))

try:
    platform = client.create_platform_component(name = "platform",
                                            hw_design = str(xsa_path),
                                            os = "standalone",
                                            cpu = "microblaze_riscv_0",
                                            domain_name = "standalone_microblaze_riscv_0",
                                            compiler = "gcc")

    comp = client.create_app_component(name="app_component",
                                    platform = "$COMPONENT_LOCATION/../platform/export/platform/platform.xpfm",
                                    domain = "standalone_microblaze_riscv_0")

    comp = client.get_component(name="app_component")
    status = comp.import_files(from_loc=str(source_dir), 
                                files=["microblaze.c",
                                       "img_pixels.inc"],
                                is_skip_copy_sources = False)
except Exception as exc:
    raise SystemExit(
        f"\nComponent creations failed. Verify paths and that the items exists where they should be.\n"
        f"\nVitis reports: {exc}.\n"
    )
