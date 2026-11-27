from pynq import Overlay, MMIO
import time
from pathlib import Path
from time import monotonic, perf_counter_ns
import numpy as np
from conv_bench import EDGE_KERNEL, conv_python, load_photo

DEFAULT_BRAM = "axi_bram_ctrl_0"
DEFAULT_GPIO = "axi_gpio_0"

# Gotta change these values to new output addresses or not hard coded
bram_O = 0x1410
bram_O_end = 0x142F

GPIO1_DATA = 0x00
GPIO1_TRI  = 0x04
GPIO2_DATA = 0x08
GPIO2_TRI  = 0x0C
OUTPUT = 0x1000 // 4

class MicroBlaze:
    def __init__(self, overlay: Overlay, bram : str, gpio : str) -> None:
        self.ol = overlay

        print("IP keys:", self.ol.ip_dict.keys())
        print("MEM keys:", self.ol.mem_dict.keys())

        self.bram_info = self.ol.mem_dict[bram]
        self.bram_base = self.bram_info["phys_addr"]
        self.bram_size = self.bram_info["addr_range"]
        self.bram = MMIO(self.bram_base, self.bram_size)
        print(f"BRAM: Address Base: {hex(self.bram_base)}, Size: {hex(self.bram_size)}")

        self.gpio_info = self.ol.ip_dict[gpio]
        self.gpio_base = self.gpio_info["phys_addr"]
        self.gpio_size = self.gpio_info["addr_range"]
        self.gpio = MMIO(self.gpio_base, self.gpio_size)
        print(f"GPIO: Address Base: {hex(self.bram_base)}, Size: {hex(self.bram_size)}")

    def free(self):
        self.ol.free()

    def setup(self):
        # ch1 output (START), ch2 output (DONE)
        self.gpio.write(GPIO1_TRI, 0x00000000)
        self.gpio.write(GPIO2_TRI, 0x00000000)
        
        # clear BRAM words before start (optional)
        self.gpio.write(GPIO1_DATA, 0)
        print("BRAM cleared:", hex(self.bram.read(0)), hex(self.bram.read(4)))    

    def start_run(self):
        self.gpio.write(GPIO1_DATA, 1)
        print("START readback:", hex(self.gpio.read(GPIO1_DATA)))
        self.gpio.write(GPIO1_DATA, 0)
        print("START readback:", hex(self.gpio.read(GPIO1_DATA)))
        
    def stop_run(self):
        self.gpio.write(GPIO1_DATA, 0)
        print("START readback:", hex(self.gpio.read(GPIO1_DATA)))

    def read_done(self):
        return self.gpio.read(GPIO2_DATA)

    def run_convolution(self):

        self.start_run()

        # poll done
        t0 = time.time()
        t_start = time.perf_counter()
        while True:
            r = self.read_done()
            if (r >> 31) & 1:
                cycles = r & 0x7FFFFFFF
                break
            if time.time() - t0 > 20.0:
                print("Timeout waiting for DONE; ch2=", hex(r))
                break
            time.sleep(0.001)

        t_done = time.perf_counter()
        elapsed = t_done - t_start

        time.sleep(0.01)
        print("BRAM data: ")
        for i in range(bram_O, bram_O_end + 1, 4):
            print(hex(i), ":", hex(self.bram.read(i)))

        self.stop_run()

        return elapsed

def load_microblaze(bitstream : str | Path):
    path = Path(bitstream).expanduser().resolve()
    if path.suffix.lower() != ".bit":
            raise ValueError(f"Select a .bit file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Bitstream not found: {path}")
    hwh = path.with_suffix(".hwh")
    if not hwh.is_file():
        raise FileNotFoundError(f"Place the matching Vivado hardware description at {hwh}")
    return MicroBlaze(Overlay(str(path), download=True), DEFAULT_BRAM, DEFAULT_GPIO)

def print_convolution(microblaze : MicroBlaze):

    time = microblaze.run_convolution()
    print("Elapsed time (seconds):", time)

def wait_status(read_status, wanted, timeout):
    deadline = monotonic() + timeout
    while True:
        status = read_status()
        if status == wanted:
            return
        if monotonic() >= deadline:
            raise TimeoutError(f"Expected GPIO DONE bit {wanted:#010x}, read {status:#010x}. "
                               "Check the GPIO setup, reset and the loaded ELF.")


def run_tiled(image, words, set_start, read_status, timeout=2.0):
    """words is the PS BRAM MMIO uint32 array. Each call owns the interface exclusively."""

    image = np.asarray(image)

    if image.ndim != 2 or min(image.shape) < 3 or image.dtype.kind not in "iu":
        raise ValueError("Expected a 2-D integer grayscale image at least 3 by 3")

    if image.min() < 0 or image.max() > 255:
        raise ValueError("Expected grayscale values in 0..255")

    if words.dtype != np.uint32 or words.size < 2048:
        raise ValueError("Map at least 8192 bytes of shared BRAM as uint32 words")

    height, width = image.shape
    edges = np.empty((height - 2, width - 2), dtype=np.int32)
    input_tile = words[:1024].reshape(32, 32)
    output_tile = words[OUTPUT:OUTPUT+900].view(np.int32).reshape(30, 30)

    set_start(0)
    wait_status(read_status, 0, timeout)

    for y in range(0, height - 2, 30):
        for x in range(0, width - 2, 30):
            tile = image[y:y+32, x:x+32]
            h, w = tile.shape
            # Only outputs whose entire 3x3 window lies inside this input are copied.
            input_tile[:h, :w] = tile
            set_start(1)
            try:
                wait_status(read_status, 1, timeout)
                oh, ow = h - 2, w - 2
                edges[y:y+oh, x:x+ow] = output_tile[:oh, :ow]
            finally:
                set_start(0)
            wait_status(read_status, 0, timeout)
    return edges


def measure_tiles(board, pixels):
    durations_ns = []
    start_ns = None

    def set_start(value):
        nonlocal start_ns
        if value == 1:
            start_ns = perf_counter_ns()
        board.gpio.write(GPIO1_DATA, value)

    def read_done():
        nonlocal start_ns
        done = (board.read_done() >> 31) & 1

        if done and start_ns is not None:
            durations_ns.append(perf_counter_ns() - start_ns)
            start_ns = None  # Record this tile only once.

        return done

    board.setup()
    begin = perf_counter_ns()
    edges = run_tiled(pixels, board.bram.array, set_start, read_done)
    total_ns = perf_counter_ns() - begin

    return edges, durations_ns, total_ns
    
def verify_microblaze(edges, photo_path, row=0, col=0, height=5, width=5):
    actual = np.asarray(edges)
    photo = load_photo(photo_path)
    shape = (len(photo) - 2, len(photo[0]) - 2)

    if actual.shape != shape:
        raise AssertionError(f"Output shape: MicroBlaze {actual.shape}, expected {shape}")

    expected = np.asarray(conv_python(photo, EDGE_KERNEL), dtype=np.int32)

    print(f"MicroBlaze shape: {actual.shape} | Python shape: {expected.shape}")

    region = np.s_[row:row + height, col:col + width]
    print(f"\nMicroBlaze output at row {row}, column {col}:")
    print(actual[region])

    print(f"\nPython output at row {row}, column {col}:")
    print(expected[region])

    mismatch = actual != expected
    count = np.count_nonzero(mismatch)
    if count:
        y, x = np.unravel_index(mismatch.argmax(), shape)
        raise AssertionError(
            f"{count:,} of {actual.size:,} pixels differ. First mismatch at "
            f"(row={y}, col={x}): MicroBlaze={actual[y, x]}, Python={expected[y, x]}"
        )
    
    print(f"\nPASS: all {actual.size:,} signed output pixels match Python ({shape[0]} x {shape[1]}).")
