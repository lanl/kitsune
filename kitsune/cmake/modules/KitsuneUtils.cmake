# ----------------------------- BEGIN MAYBE REMOVE -----------------------------
#
# Everything in this block (until END MAYBE REMOVE) is probably obsolete and, if
# so, ought to be removed. These are only used in kitsune/examples, but those
# are also likely obsolete now. But we need to double-check that before doing
# so.

#
# Get a list of all enabled tapir runtime targets so we can
# walk through each and do "stuff" (e.g., build an executable
# for each enabled target from a single source file).
#
# NOTE: This implementation assumes there is a one-to-one
# match between the compiler option flag and the name used
# in the CMake variables.  In other words,
#
#   -ftapir=TARGET_NAME
#
# has a corresponding CMAKE configuration flag:
#
#   KITSUNE_ENABLE_(TARGET_NAME)_TARGET
#
# In addition, the list '_rt_cmake_var_names' below will need
# to be updated to include any new runtime targets added to
# Kitsune+Tapir.
#
macro(get_tapir_rt_targets rt_list)
  set(_rt_cmake_var_names "OPENMP;QTHREADS;REALM;CUDA;HIP;OPENCL")
  # Always enable the serial and opencilk targets.
  list(APPEND ${rt_list} "serial")
  list(APPEND ${rt_list} "opencilk")

  foreach(rt IN ITEMS ${_rt_cmake_var_names})
    set(_enabled_var "KITSUNE_ENABLE_${rt}_TARGET")
    message(STATUS "checking for ${_enabled_var}")
    if (${_enabled_var})
      string(TOLOWER ${rt} flag)
      list(APPEND ${rt_list} ${flag})
    endif()
  endforeach()
  unset(_rt_cmake_var_names)
  unset(_kitsune_rt_flags)
endmacro()

# NOTE: the dep_list and target_libs variables used in the following
# function are expected to live in the parent scope (i.e., we're
# mucking with state outside of the function to try and debug some
# build dependency issues).
function(add_tapir_dependency target abi)
  message(STATUS "adding dependency for ${target} w/ -ftapir=${abi}")
  if (${abi} STREQUAL "opencilk")
    list(APPEND dep_list cheetah)
    set(target_libs ${target_libs} opencilk opencilk-personality-cpp PARENT_SCOPE)
    add_dependencies(${target} ${dep_list})
  elseif(${abi} STREQUAL "realm")
    list(APPEND dep_list RealmABI)
    set(target_libs ${target_libs} realm-abi realm PARENT_SCOPE)
    add_dependencies(${target} ${dep_list})
  elseif(${abi} STREQUAL "cuda")
    list(APPEND dep_list cu-abi)
    set(target_libs ${target_libs} cu-abi PARENT_SCOPE)
    add_dependencies(${target} ${dep_list})
  elseif(${abi} STREQUAL "none")
    message(STATUS "no dependencies for '-ftapir=none'...")
  elseif(${abi} STREQUAL "serial")
    message(STATUS "no dependencies for '-ftapir=serial'...")
  else()
    message(FATAL_ERROR
       "tapir dependency ${abi} not handled in add_tapir_dependency")
  endif()

endfunction()

# ------------------------------ END MAYBE REMOVE ------------------------------

# Setup a Kitsune frontend symlink (kitcc, kit++ etc.). symlink is the name of
# the frontend. Target is the actual compiler that is the target of the symlink.
macro(setup_frontend_symlink symlink target)
  # This is ugly! The create_symlink command creates a dangling symlink because
  # it is executed before clang (and perhaps flang) is built. However, if
  # everything builds correctly, it will not be dangling. Obviously, a build
  # failure will result in a dangling symlink in the build directory.
  add_custom_target(${symlink} ALL
    ${CMAKE_COMMAND} -E create_symlink
    ${target}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${symlink})

  install(FILES
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${symlink}
    DESTINATION ${CMAKE_INSTALL_BINDIR})
endmacro()
