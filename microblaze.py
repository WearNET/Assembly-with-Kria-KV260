from pynq import Overlay, MMIO
import time
from pathlib import Path

DEFAULT_BRAM = "axi_bram_ctrl_0"
DEFAULT_GPIO = "axi_gpio_0"

# Gotta change these values to new output addresses or not hard coded
bram_O = 0x1410
bram_O_end = 0x142F

GPIO1_DATA = 0x00
GPIO1_TRI  = 0x04
GPIO2_DATA = 0x08
GPIO2_TRI  = 0x0C

class MicroBlaze:
    def __init__(self, overlay: Overlay, bram : str, gpio : str) -> None:
        self.ol = overlay

        print("IP keys:", self.ol.ip_dict.keys())
        print("MEM keys:", self.ol.mem_dict.keys())

        self.bram_info = self.ol.mem_dict[bram]
        self.bram_base = self.bram_info["phys_addr"]
        self.bram_size = self.bram_info["addr_range"]
        self.bram = MMIO(self.bram_base, self.bram_size)
        print("BRAM:", hex(self.bram_base), hex(self.bram_size))

        self.gpio_info = self.ol.ip_dict[gpio]
        self.gpio_base = self.gpio_info["phys_addr"]
        self.gpio_size = self.gpio_info["addr_range"]
        self.gpio = MMIO(self.gpio_base, self.gpio_size)
        print("GPIO:", hex(self.gpio_base), hex(self.gpio_info["addr_range"]))


    def setup(self):
        # ch1 output (START), ch2 output (DONE)
        self.gpio.write(GPIO1_TRI, 0x00000000)
        self.gpio.write(GPIO2_TRI, 0x00000000)
        
        # clear BRAM words before start (optional)
        self.gpio.write(GPIO1_DATA, 0)
        print("BRAM before:", hex(self.bram.read(0)), hex(self.bram.read(4)))    

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