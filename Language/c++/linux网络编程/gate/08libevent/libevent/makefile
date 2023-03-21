
SrcFiles=event_wb.c pub.c
all:event_wb
event_wb:$(SrcFiles)
	gcc -o $@ $^ -levent 

clean:
	rm -f event_wb

.PHONY:clean all 

