CXX=arm-none-eabi-gcc
CXXFLAGS=--specs=nosys.specs -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
CXXFLAGS+=-DNDEBUG -Wall -Werror
CXXFLAGS+=-O4
CXXFLAGS+=-I${OPENCM3_DIR}/include -DSTM32F4

LDFLAGS=-L${OPENCM3_DIR}/lib -lopencm3_stm32f4 -Tstm32f411.ld -nostartfiles -Wl,--print-memory-usage

all: tfl_hello
.PHONY: flash

generated.c: ../../tfl_helloworld/model.onnx
	onnx2c $< > $@

tfl_hello: main.c generated.c
	${CXX} ${CXXFLAGS} $^ -o $@ ${LDFLAGS}

flash: tfl_hello
	openocd -f interface/stlink-v2.cfg -f target/stm32f4x.cfg -c "program tfl_hello verify reset exit"

