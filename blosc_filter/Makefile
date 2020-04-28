CC = gcc

HDF_INC=$(HDF5_ROOT)/include
CFLAGS = -I$(HDF_INC) -I$(BLOSC_ROOT)/include -shared -fPIC

TARGET = blosc_filter.so blosc_plugin.so

all: $(TARGET)

$(TARGET) : %.so: %.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f *~ $(OBJFILES) $(TARGET)
