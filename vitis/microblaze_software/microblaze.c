#include "xparameters.h"
#include "xgpio.h"
#include <stdint.h>

/* MicroBlaze data segment for axi_bram_ctrl_0 in the workshop XSA. */
#define GPIO_DEVICE_ID XPAR_XGPIO_0_BASEADDR
#define SHARED_BRAM_BASE 0xA0000000u
#define CHANNEL1 1
#define CHANNEL2 2

XGpio Gpio;

_Static_assert(sizeof(int) == 4, "Tile protocol requires 32-bit int");
static int kernel[] = {
    -1,-1,-1,
    -1, 8,-1,
    -1,-1,-1
};

// Insert the supplied row-order function from notebook Part 3 HERE.



int main(void) {
    uintptr_t base = SHARED_BRAM_BASE;
    int *input = (int *)base;
    int *output = (int *)(base + 0x1000);

    if (XGpio_Initialize(&Gpio, GPIO_DEVICE_ID) != XST_SUCCESS){
        return XST_FAILURE;
    }

    XGpio_SetDataDirection(&Gpio, CHANNEL1, 0xFFFFFFFF); // input
    XGpio_SetDataDirection(&Gpio, CHANNEL2, 0x00000000); // output

    while (1) {
        XGpio_DiscreteWrite(&Gpio, CHANNEL2, 0x0);
        
        while (!(XGpio_DiscreteRead(&Gpio, CHANNEL1) & 1u)){
            // Hold DONE until python raises START
        }

        __asm__ volatile ("fence iorw,iorw" ::: "memory");
        
        /* Insert the convolution call HERE
        Use input, output, kernel, 32 and 32 as variables. */


        __asm__ volatile ("fence iorw,iorw" ::: "memory");
        
        XGpio_DiscreteWrite(&Gpio, CHANNEL2, 0x80000000u);

        while (XGpio_DiscreteRead(&Gpio, CHANNEL1) & 1u) {
            // Hold DONE until Python finishes reading the tile and lowers START.
        }       
    }
}
