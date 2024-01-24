#include <cstdint>
#include <ATen/ATen.h>

void col_chk_enc(char transa, char transb, int64_t m, int64_t n,
                 at::Half *A, int64_t lda, int64_t stridea, 
                 at::Half * chk_v, int64_t ld_chk_v,
                 at::Half * dcolchk, int64_t ld_dcolchk,
                 int64_t num_batches);

void row_chk_enc(char transa, char transb, int64_t m, int64_t n,
                 at::Half * A, int64_t lda, int64_t stridea,
                 at::Half * chk_v, int64_t ld_chk_v,
                 at::Half * drowchk, int64_t ld_drowchk,
                 int64_t num_batches);
