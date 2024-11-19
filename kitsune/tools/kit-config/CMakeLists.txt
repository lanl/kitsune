set(LLVM_LINK_COMPONENTS
  Support)

# set(BUILDVARIABLES_SRCPATH ${CMAKE_CURRENT_SOURCE_DIR}/BuildVariables.inc.in)
# set(BUILDVARIABLES_OBJPATH ${CMAKE_CURRENT_BINARY_DIR}/BuildVariables.inc)

# Add the kit-config tool.
add_llvm_tool(kit-config
  kit-config.cpp)
