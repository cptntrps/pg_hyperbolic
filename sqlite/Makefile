CC ?= gcc
CFLAGS = -O2 -shared -fPIC -Wall
TARGET = hyp_sqlite.so

all: $(TARGET)

$(TARGET): hyp_sqlite.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean
