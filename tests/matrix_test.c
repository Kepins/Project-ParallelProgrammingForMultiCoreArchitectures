#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>
#include <math.h>
#include <stdio.h>

#include "../include/matrix.h"
#include "../include/random.h"

static Matrix m1, m2, result, correct_result;

void correct_multiplicate(Matrix a, Matrix b, Matrix c) {
  for (uint64_t i = 0; i < c.h; i++) {
    for (uint64_t j = 0; j < c.w; j++) {
      c.d[i][j] = 0.0;
      for (uint64_t k = 0; k < a.w; k++) {
        c.d[i][j] += a.d[i][k] * b.d[k][j];
      }
    }
  }
}

void assert_is_the_same(Matrix a, Matrix b) {
  CU_ASSERT_EQUAL(a.w, b.w);
  CU_ASSERT_EQUAL(a.h, b.h);

  for (uint64_t i = 0; i < a.h; i++) {
    for (uint64_t j = 0; j < a.w; j++) {
      CU_ASSERT_DOUBLE_EQUAL_FATAL(a.d[i][j], b.d[i][j], 10e-9);
    }
  }
}

void testMULTIPLICATION(void) {
  multiplicate(m1, m2, result);
  assert_is_the_same(correct_result, result);
}

void testOPENMPMULTIPLICATION(void) {
  openmp_multiplicate(m1, m2, result);
  assert_is_the_same(correct_result, result);
}

int init_suite_square_matrices(void) {
  const uint64_t MATRIX_M1_HEIGHT = 100;
  const uint64_t MATRIX_M1_WIDTH = 100;
  const uint64_t MATRIX_M2_HEIGHT = MATRIX_M1_WIDTH;
  const uint64_t MATRIX_M2_WIDTH = 100;
  const uint64_t MATRIX_RESULT_HEIGHT = MATRIX_M1_HEIGHT;
  const uint64_t MATRIX_RESULT_WIDTH = MATRIX_M2_WIDTH;

  m1 = allocate_matrix_data(MATRIX_M1_WIDTH, MATRIX_M1_HEIGHT);
  m2 = allocate_matrix_data(MATRIX_M2_WIDTH, MATRIX_M2_HEIGHT);
  result = allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);
  correct_result =
      allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);
  fill_with_random_numbers(m1);
  fill_with_random_numbers(m2);
  correct_multiplicate(m1, m2, correct_result);
  return 0;
}

int init_suite_not_square_matrices(void) {
  const uint64_t MATRIX_M1_HEIGHT = 20;
  const uint64_t MATRIX_M1_WIDTH = 100;
  const uint64_t MATRIX_M2_HEIGHT = MATRIX_M1_WIDTH;
  const uint64_t MATRIX_M2_WIDTH = 500;
  const uint64_t MATRIX_RESULT_HEIGHT = MATRIX_M1_HEIGHT;
  const uint64_t MATRIX_RESULT_WIDTH = MATRIX_M2_WIDTH;

  m1 = allocate_matrix_data(MATRIX_M1_WIDTH, MATRIX_M1_HEIGHT);
  m2 = allocate_matrix_data(MATRIX_M2_WIDTH, MATRIX_M2_HEIGHT);
  result = allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);
  correct_result =
      allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);
  fill_with_random_numbers(m1);
  fill_with_random_numbers(m2);
  correct_multiplicate(m1, m2, correct_result);
  return 0;
}

int clean_suite(void) {
  free_matrix_data(m1);
  free_matrix_data(m2);
  free_matrix_data(result);
  free_matrix_data(correct_result);

  return 0;
}

/* The main() function for setting up and running the tests.
 * Returns a CUE_SUCCESS on successful running, another
 * CUnit error code on failure.
 */
int main(void) {
  init_rand();

  CU_pSuite pSuiteSquare = NULL;
  CU_pSuite pSuiteNonSquare = NULL;

  /* initialize the CUnit test registry */
  if (CUE_SUCCESS != CU_initialize_registry())
    return CU_get_error();

  /* add a suite for square matrices to the registry */
  pSuiteSquare = CU_add_suite("Square Matrices Suite",
                              init_suite_square_matrices, clean_suite);
  if (NULL == pSuiteSquare) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* add the tests for square matrices */
  if ((NULL ==
       CU_add_test(pSuiteSquare, "test multiplicate()", testMULTIPLICATION)) ||
      (NULL == CU_add_test(pSuiteSquare, "test openmp_multiplicate()",
                           testOPENMPMULTIPLICATION))) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* add a suite for non-square matrices to the registry */
  pSuiteNonSquare = CU_add_suite("Non-Square Matrices Suite",
                                 init_suite_not_square_matrices, clean_suite);
  if (NULL == pSuiteNonSquare) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* add the tests for non-square matrices */
  if ((NULL == CU_add_test(pSuiteNonSquare, "test multiplicate()",
                           testMULTIPLICATION)) ||
      (NULL == CU_add_test(pSuiteNonSquare, "test openmp_multiplicate()",
                           testOPENMPMULTIPLICATION))) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* Run all tests using the CUnit Basic interface */
  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();
}
