MEMORY
{
	rom (rx) : ORIGIN = 0x08000000, LENGTH = 512K
	ram (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

/* Include OpenCM3 linker scripts */
INCLUDE cortex-m-generic.ld

