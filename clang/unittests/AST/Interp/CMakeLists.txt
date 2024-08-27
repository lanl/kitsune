add_clang_unittest(InterpTests
  Descriptor.cpp
  )

clang_target_link_libraries(InterpTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  LLVMTapirOpts
  )

  target_link_libraries(InterpTests
  PRIVATE
  clangTesting
)
