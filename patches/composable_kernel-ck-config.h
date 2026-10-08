/* Generated for PyTorch ROCm build — enables all CK dtypes */
#ifndef CK_CONFIG_H_IN
#define CK_CONFIG_H_IN

/* Enable all dtypes */
#define CK_ENABLE_ALL_DTYPES 1
#define CK_ENABLE_INT8 "ON"
#define CK_ENABLE_FP8 "ON"
#define CK_ENABLE_BF8 "ON"
#define CK_ENABLE_FP16 "ON"
#define CK_ENABLE_BF16 "ON"
#define CK_ENABLE_FP32 "ON"
#define CK_ENABLE_FP64 "ON"

/* WMMA supported on RDNA3/RDNA4 (gfx11xx) */
#define CK_USE_WMMA 1

/* XDL supported on CDNA (MI-series); not applicable for gfx1151 */
/* #undef CK_USE_XDL */

/* #undef CK_ENABLE_DL_KERNELS */
/* #undef CK_ENABLE_DPP_KERNELS */
/* #undef CK_USE_GFX94 */
/* #undef CK_USE_OCP_FP8 */
/* #undef CK_USE_FNUZ_FP8 */
/* #undef CK_USE_FP8_ON_UNSUPPORTED_ARCH */
/* #undef CK_USE_NATIVE_MX_SUPPORT */

#endif /* CK_CONFIG_H_IN */
