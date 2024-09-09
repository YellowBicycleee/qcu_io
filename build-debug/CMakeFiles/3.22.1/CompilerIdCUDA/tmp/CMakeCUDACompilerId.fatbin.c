#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000400,0x0000004001010002,0x00000000000003c0\n"
".quad 0x0000000000000000,0x0000005900010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000007e00be0002,0x0000000000000000,0x0000000000000350,0x0000000000000190\n"
".quad 0x0038004000590559,0x0001000700400002,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x67756265642e006f,0x2e00656d6172665f,0x676c6c61632e766e,0x766e2e0068706172\n"
".quad 0x79746f746f72702e,0x722e766e2e006570,0x6f697463612e6c65,0x747368732e00006e\n"
".quad 0x74732e0062617472,0x79732e0062617472,0x79732e006261746d,0x6e68735f6261746d\n"
".quad 0x692e766e2e007864,0x6265642e006f666e,0x656d6172665f6775,0x6c61632e766e2e00\n"
".quad 0x2e0068706172676c,0x6f746f72702e766e,0x766e2e0065707974,0x7463612e6c65722e\n"
".quad 0x00000000006e6f69,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x000500030000003f,0x0000000000000000,0x0000000000000000,0x000600030000005b\n"
".quad 0x0000000000000000,0x0000000000000000,0xffffffff00000000,0xfffffffe00000000\n"
".quad 0xfffffffd00000000,0xfffffffc00000000,0x0000000000000073,0x3605002511000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000040\n"
".quad 0x000000000000006a,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x00000000000000aa\n"
".quad 0x000000000000006a,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x0000000000000118\n"
".quad 0x0000000000000048,0x0000000300000002,0x0000000000000008,0x0000000000000018\n"
".quad 0x0000000100000032,0x0000000000000000,0x0000000000000000,0x0000000000000160\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000001,0x0000000000000000\n"
".quad 0x700000010000003f,0x0000000000000000,0x0000000000000000,0x0000000000000160\n"
".quad 0x0000000000000020,0x0000000000000003,0x0000000000000004,0x0000000000000008\n"
".quad 0x7000000b0000005b,0x0000000000000000,0x0000000000000000,0x0000000000000180\n"
".quad 0x0000000000000010,0x0000000000000000,0x0000000000000008,0x0000000000000008\n"
".quad 0x0000000500000006,0x0000000000000350,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000070,0x0000000000000070,0x0000000000000008,0x0000000500000001\n"
".quad 0x0000000000000350,0x0000000000000000,0x0000000000000000,0x0000000000000070\n"
".quad 0x0000000000000070, 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[130];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif