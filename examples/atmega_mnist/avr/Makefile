# Use arduino-mk to compile libraries using the Arduino framework.
BOARD_TAG    = uno
ARDUINO_INSTALL?=/usr/share/arduino/
include $(ARDUINO_INSTALL)/Arduino.mk


TOUCH_OBJ = build-$(BOARD_TAG)/TouchScreen.o
TFT_OBJ = build-$(BOARD_TAG)/TFT.o
GFX_OBJ = build-$(BOARD_TAG)/GFX.o

CXXFLAGS += -I Adafruit_TouchScreen
CXXFLAGS += -I Adafruit-GFX-Library
CXXFLAGS += -I MCUFRIEND_kbv
CXXFLAGS += -D USE_ADAFRUIT_SHIELD_PINOUT=1
OTHER_OBJS = $(TOUCH_OBJ) $(TFT_OBJ) $(GFX_OBJ) build-$(BOARD_TAG)/libcore.a

$(TARGET_ELF): $(OTHER_OBJS)

$(TOUCH_OBJ): Adafruit_TouchScreen/TouchScreen.cpp
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) Adafruit_TouchScreen/TouchScreen.cpp -o $(TOUCH_OBJ)

$(TFT_OBJ): MCUFRIEND_kbv/MCUFRIEND_kbv.cpp
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) MCUFRIEND_kbv/MCUFRIEND_kbv.cpp -o $(TFT_OBJ)

$(GFX_OBJ): Adafruit-GFX-Library/Adafruit_GFX.cpp
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) Adafruit-GFX-Library/Adafruit_GFX.cpp -o $(GFX_OBJ)


.PHONY: get_libs flash

flash: $(TARGET_ELF)
	avrdude -v -p atmega328p -c arduino -P/dev/ttyACM0  -U flash:w:$(TARGET_ELF):e

get_libs:
	git clone --depth=1 https://github.com/adafruit/Adafruit_TouchScreen.git
	git clone --depth=1 https://github.com/adafruit/Adafruit-GFX-Library.git
	git clone --depth=1 https://github.com/prenticedavid/MCUFRIEND_kbv.git

