import concurrent.futures
import itertools
import subprocess

BLOCK_SIZE_INIT = [128, 256, 512, 1024]
BLOCK_SIZE_C_XXX = [128, 256, 512, 1024]
BLOCK_SIZE_QUANTILE_BIASES = [128, 256, 512, 1024]
BLOCK_SIZE_PPV_PX = [128, 256, 512, 1024]

OPTIONS: list[tuple[int, int, int, int]] = list(
    itertools.product(BLOCK_SIZE_INIT, BLOCK_SIZE_C_XXX, BLOCK_SIZE_QUANTILE_BIASES, BLOCK_SIZE_PPV_PX))

COMMAND = [
    "nvcc",
    "--expt-relaxed-constexpr",
    "-use_fast_math",
    "-Xptxas",
    "-O3",
    "-Xcompiler",
    "-O3",
    "-arch=sm_86",
    "--forward-unknown-to-host-compiler",
    "-O3",
    "-mtune=native",
    "-march=native",
    "-std=c++20",
]


def options_to_command(options: tuple[int, int, int, int], cuda_file: str, out_folder: str) -> list[str]:
    out_name = cuda_file[::-1].split('.', maxsplit=1)[1][::-1]
    return [
        *COMMAND,
        "-D_BLOCK_SIZE_INIT=" + str(options[0]),
        "-D_BLOCK_SIZE_C_XXX=" + str(options[1]),
        "-D_BLOCK_SIZE_QUANTILE_BIASES=" + str(options[2]),
        "-D_BLOCK_SIZE_PPV_PX=" + str(options[3]),
        cuda_file,
        "-o",
        f"{out_folder}/{out_name}_{options[0]}_{options[1]}_{options[2]}_{options[3]}.out"
    ]


def run_command(value: tuple[int, int, int, int]):
    subprocess.run(
        options_to_command(value, "main.cu", "compiled"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        check=True,
    )


def main():
    print("BLOCK_SIZE_INIT BLOCK_SIZE_C_XXX BLOCK_SIZE_QUANTILE_BIASES BLOCK_SIZE_PPV_PX")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_command, value): value for value in OPTIONS}
        for future in concurrent.futures.as_completed(futures):
            value = futures[future]
            try:
                print(f"{value} ✅")
            except Exception as exc:
                print(f"{value} ERROR {exc}")


if __name__ == "__main__":
    main()
