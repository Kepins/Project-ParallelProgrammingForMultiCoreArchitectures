#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>
#include <math.h>
#include <stdio.h>

#include "../include/matrix.h"
#include "../include/random.h"

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
      CU_ASSERT_DOUBLE_EQUAL(a.d[i][j], b.d[i][j], 10e-9);
    }
  }
}

void testMULTIPLICATION_SQUARE(void) {
  const uint64_t MATRIX_M1_HEIGHT = 10;
  const uint64_t MATRIX_M1_WIDTH = 10;
  const uint64_t MATRIX_M2_HEIGHT = MATRIX_M1_WIDTH;
  const uint64_t MATRIX_M2_WIDTH = 10;
  const uint64_t MATRIX_RESULT_HEIGHT = MATRIX_M1_HEIGHT;
  const uint64_t MATRIX_RESULT_WIDTH = MATRIX_M2_WIDTH;

  Matrix m1 = allocate_matrix_data(MATRIX_M1_WIDTH, MATRIX_M1_HEIGHT);
  Matrix m2 = allocate_matrix_data(MATRIX_M2_WIDTH, MATRIX_M2_HEIGHT);
  Matrix result =
      allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);
  Matrix correct_result =
      allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);

  fill_with_random_numbers(m1);
  fill_with_random_numbers(m2);

  correct_multiplicate(m1, m2, correct_result);
  multiplicate(m1, m2, result);

  assert_is_the_same(correct_result, result);

  free_matrix_data(m1);
  free_matrix_data(m2);
  free_matrix_data(result);
  free_matrix_data(correct_result);
}

void testMULTIPLICATION_NOT_SQUARE(void) {
  const uint64_t MATRIX_M1_HEIGHT = 2;
  const uint64_t MATRIX_M1_WIDTH = 10;
  const uint64_t MATRIX_M2_HEIGHT = MATRIX_M1_WIDTH;
  const uint64_t MATRIX_M2_WIDTH = 50;
  const uint64_t MATRIX_RESULT_HEIGHT = MATRIX_M1_HEIGHT;
  const uint64_t MATRIX_RESULT_WIDTH = MATRIX_M2_WIDTH;

  Matrix m1 = allocate_matrix_data(MATRIX_M1_WIDTH, MATRIX_M1_HEIGHT);
  Matrix m2 = allocate_matrix_data(MATRIX_M2_WIDTH, MATRIX_M2_HEIGHT);
  Matrix result =
      allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);
  Matrix correct_result =
      allocate_matrix_data(MATRIX_RESULT_WIDTH, MATRIX_RESULT_HEIGHT);

  fill_with_random_numbers(m1);
  fill_with_random_numbers(m2);

  correct_multiplicate(m1, m2, correct_result);
  multiplicate(m1, m2, result);

  assert_is_the_same(correct_result, result);

  free_matrix_data(m1);
  free_matrix_data(m2);
  free_matrix_data(result);
  free_matrix_data(correct_result);
}

int init_suite(void) {
  init_rand();
  return 0;
}

int clean_suite(void) { return 0; }

/* The main() function for setting up and running the tests.
 * Returns a CUE_SUCCESS on successful running, another
 * CUnit error code on failure.
 */
int main(void) {
  CU_pSuite pSuite = NULL;

  /* initialize the CUnit test registry */
  if (CUE_SUCCESS != CU_initialize_registry())
    return CU_get_error();

  /* add a suite to the registry */
  pSuite = CU_add_suite("Suite", init_suite, clean_suite);
  if (NULL == pSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* add the tests to the suite */
  if ((NULL == CU_add_test(pSuite, "test multiplicate() with square matricies",
                           testMULTIPLICATION_SQUARE)) ||
      (NULL == CU_add_test(pSuite,
                           "test multiplicate() with not square matricies",
                           testMULTIPLICATION_NOT_SQUARE))) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* Run all tests using the CUnit Basic interface */
  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();
}
