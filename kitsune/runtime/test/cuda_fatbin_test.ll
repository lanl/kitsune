; ModuleID = 'cuda_fatbin_test.cpp'
source_filename = "cuda_fatbin_test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque
%struct.stat = type { i64, i64, i64, i32, i32, i32, i32, i64, i64, i64, i64, %struct.timespec, %struct.timespec, %struct.timespec, [3 x i64] }
%struct.timespec = type { i64, i64 }
%struct.CUctx_st = type opaque
%struct.CUstream_st = type opaque
%struct.CUmod_st = type opaque
%struct.CUfunc_st = type opaque

@fatbin = dso_local global i8* null, align 8
@.str = private unnamed_addr constant [12 x i8] c"vadd.fatbin\00", align 1
@FATBIN_FILE = dso_local global i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str, i32 0, i32 0), align 8
@.str.1 = private unnamed_addr constant [3 x i8] c"rb\00", align 1
@.str.2 = private unnamed_addr constant [32 x i8] c"failed to open fat binary file!\00", align 1
@.str.3 = private unnamed_addr constant [40 x i8] c"fp && \22failed to open fat binary file!\22\00", align 1
@.str.4 = private unnamed_addr constant [21 x i8] c"cuda_fatbin_test.cpp\00", align 1
@__PRETTY_FUNCTION__._Z13readFatBinaryPKc = private unnamed_addr constant [34 x i8] c"void *readFatBinary(const char *)\00", align 1
@stderr = external global %struct._IO_FILE*, align 8
@.str.5 = private unnamed_addr constant [32 x i8] c"error reading fat binary file!\0A\00", align 1
@.str.6 = private unnamed_addr constant [32 x i8] c"kitrt: %s failed with error %s\0A\00", align 1
@.str.7 = private unnamed_addr constant [10 x i8] c"cuInit(0)\00", align 1
@.str.8 = private unnamed_addr constant [24 x i8] c"cuDeviceGet(&device, 0)\00", align 1
@.str.9 = private unnamed_addr constant [33 x i8] c"cuCtxCreate(&context, 0, device)\00", align 1
@.str.10 = private unnamed_addr constant [34 x i8] c"cuModuleLoadData(&module, fatbin)\00", align 1
@.str.11 = private unnamed_addr constant [14 x i8] c"VecAdd_kernel\00", align 1
@.str.12 = private unnamed_addr constant [53 x i8] c"cuModuleGetFunction(&kfunc, module, \22VecAdd_kernel\22)\00", align 1
@.str.13 = private unnamed_addr constant [66 x i8] c"cuMemAllocManaged(&a_data, sizeof(float) * N, CU_MEM_ATTACH_HOST)\00", align 1
@.str.14 = private unnamed_addr constant [66 x i8] c"cuMemAllocManaged(&b_data, sizeof(float) * N, CU_MEM_ATTACH_HOST)\00", align 1
@.str.15 = private unnamed_addr constant [66 x i8] c"cuMemAllocManaged(&c_data, sizeof(float) * N, CU_MEM_ATTACH_HOST)\00", align 1
@.str.16 = private unnamed_addr constant [27 x i8] c"cuStreamCreate(&stream, 0)\00", align 1
@.str.17 = private unnamed_addr constant [89 x i8] c"cuLaunchKernel(kfunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, stream, args, NULL)\00", align 1
@.str.18 = private unnamed_addr constant [28 x i8] c"cuStreamSynchronize(stream)\00", align 1
@.str.19 = private unnamed_addr constant [18 x i8] c"error count = %d\0A\00", align 1

; Function Attrs: mustprogress noinline optnone sspstrong uwtable
define dso_local i8* @_Z13readFatBinaryPKc(i8* %0) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.stat, align 8
  %6 = alloca i64, align 8
  %7 = alloca %struct._IO_FILE*, align 8
  store i8* %0, i8** %3, align 8
  store i8* null, i8** %4, align 8
  %8 = load i8*, i8** %3, align 8
  %9 = call i32 @stat(i8* %8, %struct.stat* %5) #5
  %10 = getelementptr inbounds %struct.stat, %struct.stat* %5, i32 0, i32 8
  %11 = load i64, i64* %10, align 8
  store i64 %11, i64* %6, align 8
  %12 = load i64, i64* %6, align 8
  %13 = mul i64 1, %12
  %14 = call noalias align 16 i8* @malloc(i64 %13) #5
  store i8* %14, i8** %4, align 8
  %15 = load i8*, i8** @FATBIN_FILE, align 8
  %16 = call noalias %struct._IO_FILE* @fopen(i8* %15, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i64 0, i64 0))
  store %struct._IO_FILE* %16, %struct._IO_FILE** %7, align 8
  %17 = load %struct._IO_FILE*, %struct._IO_FILE** %7, align 8
  %18 = icmp ne %struct._IO_FILE* %17, null
  br i1 %18, label %19, label %20

19:                                               ; preds = %1
  br label %20

20:                                               ; preds = %19, %1
  %21 = phi i1 [ false, %1 ], [ true, %19 ]
  br i1 %21, label %22, label %23

22:                                               ; preds = %20
  br label %25

23:                                               ; preds = %20
  call void @__assert_fail(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.3, i64 0, i64 0), i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.4, i64 0, i64 0), i32 31, i8* getelementptr inbounds ([34 x i8], [34 x i8]* @__PRETTY_FUNCTION__._Z13readFatBinaryPKc, i64 0, i64 0)) #6
  unreachable

24:                                               ; No predecessors!
  br label %25

25:                                               ; preds = %24, %22
  %26 = load i8*, i8** %4, align 8
  %27 = load i64, i64* %6, align 8
  %28 = load %struct._IO_FILE*, %struct._IO_FILE** %7, align 8
  %29 = call i64 @fread(i8* %26, i64 1, i64 %27, %struct._IO_FILE* %28)
  %30 = load i64, i64* %6, align 8
  %31 = icmp ne i64 %29, %30
  br i1 %31, label %32, label %36

32:                                               ; preds = %25
  %33 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %34 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %33, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.5, i64 0, i64 0))
  %35 = load i8*, i8** %4, align 8
  call void @free(i8* %35) #5
  store i8* null, i8** %2, align 8
  br label %40

36:                                               ; preds = %25
  %37 = load %struct._IO_FILE*, %struct._IO_FILE** %7, align 8
  %38 = call i32 @fclose(%struct._IO_FILE* %37)
  %39 = load i8*, i8** %4, align 8
  store i8* %39, i8** %2, align 8
  br label %40

40:                                               ; preds = %36, %32
  %41 = load i8*, i8** %2, align 8
  ret i8* %41
}

; Function Attrs: nounwind
declare i32 @stat(i8*, %struct.stat*) #1

; Function Attrs: nounwind
declare noalias align 16 i8* @malloc(i64) #1

declare noalias %struct._IO_FILE* @fopen(i8*, i8*) #2

; Function Attrs: noreturn nounwind
declare void @__assert_fail(i8*, i8*, i32, i8*) #3

declare i64 @fread(i8*, i64, i64, %struct._IO_FILE*) #2

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #2

; Function Attrs: nounwind
declare void @free(i8*) #1

declare i32 @fclose(%struct._IO_FILE*) #2

; Function Attrs: mustprogress noinline norecurse optnone sspstrong uwtable
define dso_local i32 @main(i32 %0, i8** %1) #4 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca %struct.CUctx_st*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %struct.CUstream_st*, align 8
  %9 = alloca i32, align 4
  %10 = alloca i8*, align 8
  %11 = alloca i32, align 4
  %12 = alloca i8*, align 8
  %13 = alloca i32, align 4
  %14 = alloca i8*, align 8
  %15 = alloca %struct.CUmod_st*, align 8
  %16 = alloca i32, align 4
  %17 = alloca i8*, align 8
  %18 = alloca %struct.CUfunc_st*, align 8
  %19 = alloca i32, align 4
  %20 = alloca i8*, align 8
  %21 = alloca i64, align 8
  %22 = alloca i64, align 8
  %23 = alloca i64, align 8
  %24 = alloca i64, align 8
  %25 = alloca i32, align 4
  %26 = alloca i8*, align 8
  %27 = alloca i32, align 4
  %28 = alloca i8*, align 8
  %29 = alloca i32, align 4
  %30 = alloca i8*, align 8
  %31 = alloca float*, align 8
  %32 = alloca float*, align 8
  %33 = alloca float*, align 8
  %34 = alloca i64, align 8
  %35 = alloca i32, align 4
  %36 = alloca i8*, align 8
  %37 = alloca i32, align 4
  %38 = alloca i32, align 4
  %39 = alloca [4 x i8*], align 16
  %40 = alloca i32, align 4
  %41 = alloca i8*, align 8
  %42 = alloca i32, align 4
  %43 = alloca i8*, align 8
  %44 = alloca i32, align 4
  %45 = alloca i64, align 8
  %46 = alloca float, align 4
  store i32 0, i32* %3, align 4
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  %47 = load i8*, i8** @FATBIN_FILE, align 8
  %48 = call i8* @_Z13readFatBinaryPKc(i8* %47)
  store i8* %48, i8** @fatbin, align 8
  %49 = load i8*, i8** @fatbin, align 8
  %50 = icmp ne i8* %49, null
  br i1 %50, label %52, label %51

51:                                               ; preds = %2
  store i32 1, i32* %3, align 4
  br label %269

52:                                               ; preds = %2
  br label %53

53:                                               ; preds = %52
  %54 = call i32 @cuInit(i32 0)
  store i32 %54, i32* %9, align 4
  %55 = load i32, i32* %9, align 4
  %56 = icmp ne i32 %55, 0
  br i1 %56, label %57, label %63

57:                                               ; preds = %53
  %58 = load i32, i32* %9, align 4
  %59 = call i32 @cuGetErrorName(i32 %58, i8** %10)
  %60 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %61 = load i8*, i8** %10, align 8
  %62 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %60, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.7, i64 0, i64 0), i8* %61)
  call void @exit(i32 1) #6
  unreachable

63:                                               ; preds = %53
  br label %64

64:                                               ; preds = %63
  br label %65

65:                                               ; preds = %64
  %66 = call i32 @cuDeviceGet(i32* %7, i32 0)
  store i32 %66, i32* %11, align 4
  %67 = load i32, i32* %11, align 4
  %68 = icmp ne i32 %67, 0
  br i1 %68, label %69, label %75

69:                                               ; preds = %65
  %70 = load i32, i32* %11, align 4
  %71 = call i32 @cuGetErrorName(i32 %70, i8** %12)
  %72 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %73 = load i8*, i8** %12, align 8
  %74 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %72, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.8, i64 0, i64 0), i8* %73)
  call void @exit(i32 1) #6
  unreachable

75:                                               ; preds = %65
  br label %76

76:                                               ; preds = %75
  br label %77

77:                                               ; preds = %76
  %78 = load i32, i32* %7, align 4
  %79 = call i32 @cuCtxCreate_v2(%struct.CUctx_st** %6, i32 0, i32 %78)
  store i32 %79, i32* %13, align 4
  %80 = load i32, i32* %13, align 4
  %81 = icmp ne i32 %80, 0
  br i1 %81, label %82, label %88

82:                                               ; preds = %77
  %83 = load i32, i32* %13, align 4
  %84 = call i32 @cuGetErrorName(i32 %83, i8** %14)
  %85 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %86 = load i8*, i8** %14, align 8
  %87 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %85, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.9, i64 0, i64 0), i8* %86)
  call void @exit(i32 1) #6
  unreachable

88:                                               ; preds = %77
  br label %89

89:                                               ; preds = %88
  br label %90

90:                                               ; preds = %89
  %91 = load i8*, i8** @fatbin, align 8
  %92 = call i32 @cuModuleLoadData(%struct.CUmod_st** %15, i8* %91)
  store i32 %92, i32* %16, align 4
  %93 = load i32, i32* %16, align 4
  %94 = icmp ne i32 %93, 0
  br i1 %94, label %95, label %101

95:                                               ; preds = %90
  %96 = load i32, i32* %16, align 4
  %97 = call i32 @cuGetErrorName(i32 %96, i8** %17)
  %98 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %99 = load i8*, i8** %17, align 8
  %100 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %98, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.10, i64 0, i64 0), i8* %99)
  call void @exit(i32 1) #6
  unreachable

101:                                              ; preds = %90
  br label %102

102:                                              ; preds = %101
  br label %103

103:                                              ; preds = %102
  %104 = load %struct.CUmod_st*, %struct.CUmod_st** %15, align 8
  %105 = call i32 @cuModuleGetFunction(%struct.CUfunc_st** %18, %struct.CUmod_st* %104, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.11, i64 0, i64 0))
  store i32 %105, i32* %19, align 4
  %106 = load i32, i32* %19, align 4
  %107 = icmp ne i32 %106, 0
  br i1 %107, label %108, label %114

108:                                              ; preds = %103
  %109 = load i32, i32* %19, align 4
  %110 = call i32 @cuGetErrorName(i32 %109, i8** %20)
  %111 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %112 = load i8*, i8** %20, align 8
  %113 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %111, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.12, i64 0, i64 0), i8* %112)
  call void @exit(i32 1) #6
  unreachable

114:                                              ; preds = %103
  br label %115

115:                                              ; preds = %114
  store i64 1024, i64* %21, align 8
  br label %116

116:                                              ; preds = %115
  %117 = call i32 @cuMemAllocManaged(i64* %22, i64 4096, i32 2)
  store i32 %117, i32* %25, align 4
  %118 = load i32, i32* %25, align 4
  %119 = icmp ne i32 %118, 0
  br i1 %119, label %120, label %126

120:                                              ; preds = %116
  %121 = load i32, i32* %25, align 4
  %122 = call i32 @cuGetErrorName(i32 %121, i8** %26)
  %123 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %124 = load i8*, i8** %26, align 8
  %125 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %123, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([66 x i8], [66 x i8]* @.str.13, i64 0, i64 0), i8* %124)
  call void @exit(i32 1) #6
  unreachable

126:                                              ; preds = %116
  br label %127

127:                                              ; preds = %126
  br label %128

128:                                              ; preds = %127
  %129 = call i32 @cuMemAllocManaged(i64* %23, i64 4096, i32 2)
  store i32 %129, i32* %27, align 4
  %130 = load i32, i32* %27, align 4
  %131 = icmp ne i32 %130, 0
  br i1 %131, label %132, label %138

132:                                              ; preds = %128
  %133 = load i32, i32* %27, align 4
  %134 = call i32 @cuGetErrorName(i32 %133, i8** %28)
  %135 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %136 = load i8*, i8** %28, align 8
  %137 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %135, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([66 x i8], [66 x i8]* @.str.14, i64 0, i64 0), i8* %136)
  call void @exit(i32 1) #6
  unreachable

138:                                              ; preds = %128
  br label %139

139:                                              ; preds = %138
  br label %140

140:                                              ; preds = %139
  %141 = call i32 @cuMemAllocManaged(i64* %24, i64 4096, i32 2)
  store i32 %141, i32* %29, align 4
  %142 = load i32, i32* %29, align 4
  %143 = icmp ne i32 %142, 0
  br i1 %143, label %144, label %150

144:                                              ; preds = %140
  %145 = load i32, i32* %29, align 4
  %146 = call i32 @cuGetErrorName(i32 %145, i8** %30)
  %147 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %148 = load i8*, i8** %30, align 8
  %149 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %147, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([66 x i8], [66 x i8]* @.str.15, i64 0, i64 0), i8* %148)
  call void @exit(i32 1) #6
  unreachable

150:                                              ; preds = %140
  br label %151

151:                                              ; preds = %150
  %152 = load i64, i64* %22, align 8
  %153 = inttoptr i64 %152 to float*
  store float* %153, float** %31, align 8
  %154 = load i64, i64* %23, align 8
  %155 = inttoptr i64 %154 to float*
  store float* %155, float** %32, align 8
  %156 = load i64, i64* %24, align 8
  %157 = inttoptr i64 %156 to float*
  store float* %157, float** %33, align 8
  store i64 0, i64* %34, align 8
  br label %158

158:                                              ; preds = %178, %151
  %159 = load i64, i64* %34, align 8
  %160 = icmp ult i64 %159, 1024
  br i1 %160, label %161, label %181

161:                                              ; preds = %158
  %162 = load i64, i64* %34, align 8
  %163 = uitofp i64 %162 to float
  %164 = load float*, float** %31, align 8
  %165 = load i64, i64* %34, align 8
  %166 = getelementptr inbounds float, float* %164, i64 %165
  store float %163, float* %166, align 4
  %167 = load float*, float** %31, align 8
  %168 = load i64, i64* %34, align 8
  %169 = getelementptr inbounds float, float* %167, i64 %168
  %170 = load float, float* %169, align 4
  %171 = fadd float %170, 1.000000e+00
  %172 = load float*, float** %32, align 8
  %173 = load i64, i64* %34, align 8
  %174 = getelementptr inbounds float, float* %172, i64 %173
  store float %171, float* %174, align 4
  %175 = load float*, float** %33, align 8
  %176 = load i64, i64* %34, align 8
  %177 = getelementptr inbounds float, float* %175, i64 %176
  store float 0.000000e+00, float* %177, align 4
  br label %178

178:                                              ; preds = %161
  %179 = load i64, i64* %34, align 8
  %180 = add i64 %179, 1
  store i64 %180, i64* %34, align 8
  br label %158, !llvm.loop !6

181:                                              ; preds = %158
  br label %182

182:                                              ; preds = %181
  %183 = call i32 @cuStreamCreate(%struct.CUstream_st** %8, i32 0)
  store i32 %183, i32* %35, align 4
  %184 = load i32, i32* %35, align 4
  %185 = icmp ne i32 %184, 0
  br i1 %185, label %186, label %192

186:                                              ; preds = %182
  %187 = load i32, i32* %35, align 4
  %188 = call i32 @cuGetErrorName(i32 %187, i8** %36)
  %189 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %190 = load i8*, i8** %36, align 8
  %191 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %189, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.16, i64 0, i64 0), i8* %190)
  call void @exit(i32 1) #6
  unreachable

192:                                              ; preds = %182
  br label %193

193:                                              ; preds = %192
  store i32 256, i32* %37, align 4
  %194 = load i32, i32* %37, align 4
  %195 = sext i32 %194 to i64
  %196 = add i64 1024, %195
  %197 = sub i64 %196, 1
  %198 = load i32, i32* %37, align 4
  %199 = sext i32 %198 to i64
  %200 = udiv i64 %197, %199
  %201 = trunc i64 %200 to i32
  store i32 %201, i32* %38, align 4
  %202 = getelementptr inbounds [4 x i8*], [4 x i8*]* %39, i64 0, i64 0
  %203 = bitcast float** %31 to i8*
  store i8* %203, i8** %202, align 8
  %204 = getelementptr inbounds i8*, i8** %202, i64 1
  %205 = bitcast float** %32 to i8*
  store i8* %205, i8** %204, align 8
  %206 = getelementptr inbounds i8*, i8** %204, i64 1
  %207 = bitcast float** %33 to i8*
  store i8* %207, i8** %206, align 8
  %208 = getelementptr inbounds i8*, i8** %206, i64 1
  %209 = bitcast i64* %21 to i8*
  store i8* %209, i8** %208, align 8
  br label %210

210:                                              ; preds = %193
  %211 = load %struct.CUfunc_st*, %struct.CUfunc_st** %18, align 8
  %212 = load i32, i32* %38, align 4
  %213 = load i32, i32* %37, align 4
  %214 = load %struct.CUstream_st*, %struct.CUstream_st** %8, align 8
  %215 = getelementptr inbounds [4 x i8*], [4 x i8*]* %39, i64 0, i64 0
  %216 = call i32 @cuLaunchKernel(%struct.CUfunc_st* %211, i32 %212, i32 1, i32 1, i32 %213, i32 1, i32 1, i32 0, %struct.CUstream_st* %214, i8** %215, i8** null)
  store i32 %216, i32* %40, align 4
  %217 = load i32, i32* %40, align 4
  %218 = icmp ne i32 %217, 0
  br i1 %218, label %219, label %225

219:                                              ; preds = %210
  %220 = load i32, i32* %40, align 4
  %221 = call i32 @cuGetErrorName(i32 %220, i8** %41)
  %222 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %223 = load i8*, i8** %41, align 8
  %224 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %222, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([89 x i8], [89 x i8]* @.str.17, i64 0, i64 0), i8* %223)
  call void @exit(i32 1) #6
  unreachable

225:                                              ; preds = %210
  br label %226

226:                                              ; preds = %225
  br label %227

227:                                              ; preds = %226
  %228 = load %struct.CUstream_st*, %struct.CUstream_st** %8, align 8
  %229 = call i32 @cuStreamSynchronize(%struct.CUstream_st* %228)
  store i32 %229, i32* %42, align 4
  %230 = load i32, i32* %42, align 4
  %231 = icmp ne i32 %230, 0
  br i1 %231, label %232, label %238

232:                                              ; preds = %227
  %233 = load i32, i32* %42, align 4
  %234 = call i32 @cuGetErrorName(i32 %233, i8** %43)
  %235 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %236 = load i8*, i8** %43, align 8
  %237 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %235, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.18, i64 0, i64 0), i8* %236)
  call void @exit(i32 1) #6
  unreachable

238:                                              ; preds = %227
  br label %239

239:                                              ; preds = %238
  store i32 0, i32* %44, align 4
  store i64 0, i64* %45, align 8
  br label %240

240:                                              ; preds = %263, %239
  %241 = load i64, i64* %45, align 8
  %242 = icmp ult i64 %241, 1024
  br i1 %242, label %243, label %266

243:                                              ; preds = %240
  %244 = load float*, float** %31, align 8
  %245 = load i64, i64* %45, align 8
  %246 = getelementptr inbounds float, float* %244, i64 %245
  %247 = load float, float* %246, align 4
  %248 = load float*, float** %32, align 8
  %249 = load i64, i64* %45, align 8
  %250 = getelementptr inbounds float, float* %248, i64 %249
  %251 = load float, float* %250, align 4
  %252 = fadd float %247, %251
  store float %252, float* %46, align 4
  %253 = load float, float* %46, align 4
  %254 = load float*, float** %33, align 8
  %255 = load i64, i64* %45, align 8
  %256 = getelementptr inbounds float, float* %254, i64 %255
  %257 = load float, float* %256, align 4
  %258 = fcmp une float %253, %257
  br i1 %258, label %259, label %262

259:                                              ; preds = %243
  %260 = load i32, i32* %44, align 4
  %261 = add i32 %260, 1
  store i32 %261, i32* %44, align 4
  br label %262

262:                                              ; preds = %259, %243
  br label %263

263:                                              ; preds = %262
  %264 = load i64, i64* %45, align 8
  %265 = add i64 %264, 1
  store i64 %265, i64* %45, align 8
  br label %240, !llvm.loop !8

266:                                              ; preds = %240
  %267 = load i32, i32* %44, align 4
  %268 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.19, i64 0, i64 0), i32 %267)
  store i32 0, i32* %3, align 4
  br label %269

269:                                              ; preds = %266, %51
  %270 = load i32, i32* %3, align 4
  ret i32 %270
}

declare i32 @cuInit(i32) #2

declare i32 @cuGetErrorName(i32, i8**) #2

; Function Attrs: noreturn nounwind
declare void @exit(i32) #3

declare i32 @cuDeviceGet(i32*, i32) #2

declare i32 @cuCtxCreate_v2(%struct.CUctx_st**, i32, i32) #2

declare i32 @cuModuleLoadData(%struct.CUmod_st**, i8*) #2

declare i32 @cuModuleGetFunction(%struct.CUfunc_st**, %struct.CUmod_st*, i8*) #2

declare i32 @cuMemAllocManaged(i64*, i64, i32) #2

declare i32 @cuStreamCreate(%struct.CUstream_st**, i32) #2

declare i32 @cuLaunchKernel(%struct.CUfunc_st*, i32, i32, i32, i32, i32, i32, i32, %struct.CUstream_st*, i8**, i8**) #2

declare i32 @cuStreamSynchronize(%struct.CUstream_st*) #2

declare i32 @printf(i8*, ...) #2

attributes #0 = { mustprogress noinline optnone sspstrong uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress noinline norecurse optnone sspstrong uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 13.0.1"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
