target_sources(unit_tests
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_heat_equation.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_wave_equation.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_sG.cpp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/test_coupled_function.cpp
)
