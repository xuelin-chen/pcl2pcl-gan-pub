nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib =  /usr/local/cuda-8.0/lib64
tensorflow_inc = /usr/local/lib/python3.5/dist-packages/tensorflow/include
tensorflow_lib = /usr/local/lib/python3.5/dist-packages/tensorflow
nsync = /usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow_inc) -I $(nsync) -lcudart -L $(cudalib) -L $(tensorflow_lib) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow_inc) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow_inc) -I $(nsync) -lcudart -L $(cudalib) -L $(tensorflow_lib) -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow_inc) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
