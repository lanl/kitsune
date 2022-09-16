%.kit.pdf : %.cpp
	-@rm -rf $<-cfg-tmp
	${clang} -fkokkos -fkokkos-no-init -ftapir=nvcuda -c -emit-llvm -o $<.bc $<
	-@mkdir -p $<-cfg-tmp
	${opt} -O3 --dot-cfg -cfg-dot-filename-prefix=$<-cfg-tmp/$< $<.bc 2> /dev/null
	dot -Tpdf -o $@ $<-cfg-tmp/$<.main.outline_.*.dot
	@-rm -rf $@-cfg-tmp
