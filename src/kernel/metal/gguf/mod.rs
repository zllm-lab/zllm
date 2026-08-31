//! GGUF/GGML 量化 matmul:gemv/gemm 家族、gated 变体与 fused GEMM。

mod dispatch;
pub mod experts;
pub use dispatch::*;
pub use experts::*;

const BASE_SHADERS: &str = r#"
// IQ3_XXS 码本(256 条,每条打包 4 个字节)与符号表(128 条)。来自 ggml-common.h。
constant uint iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414, 0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404, 0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34, 0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c, 0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434, 0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e, 0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24, 0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414, 0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c, 0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};
constant uchar ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
// IQ4_NL/IQ4_XS 码本(16 个 int8 值)。来自 ggml-common.h。
constant char kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
// IQ3_S 码本(512 条 u32)。来自 ggml-common.h。
constant uint iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305, 0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905, 0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09, 0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b, 0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b, 0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d, 0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03, 0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03, 0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901, 0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d, 0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303, 0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501, 0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105, 0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505, 0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707, 0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b, 0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01, 0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f, 0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305, 0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103, 0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509, 0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b, 0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f, 0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f, 0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f, 0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109, 0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f, 0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509, 0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303, 0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f, 0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907, 0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703, 0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03, 0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01, 0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01, 0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505, 0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b, 0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107, 0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509, 0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303, 0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103, 0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05, 0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f, 0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701, 0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909, 0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305, 0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d, 0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b, 0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d, 0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09, 0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309, 0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709, 0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f, 0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303, 0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503, 0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b, 0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// IQ2_XS 码本(512 条 ulong)。来自 ggml-common.h。
constant ulong iq2xs_grid[512] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b, 0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919, 0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b, 0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919, 0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819, 0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b, 0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908, 0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908, 0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808, 0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08, 0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808, 0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819, 0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908, 0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819, 0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808, 0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b, 0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819, 0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819, 0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808, 0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908, 0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b, 0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819, 0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908, 0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808, 0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819, 0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808, 0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808, 0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808, 0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908, 0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808, 0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b, 0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819, 0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908, 0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808, 0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919, 0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19, 0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b, 0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b, 0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808, 0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08, 0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808, 0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808, 0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819, 0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919, 0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808, 0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819, 0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819, 0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908, 0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908, 0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b, 0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908, 0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908, 0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808, 0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819, 0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808, 0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b, 0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819, 0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819, 0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08, 0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808, 0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19, 0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919, 0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b, 0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808, 0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b, 0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b, 0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808, 0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819, 0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808, 0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808, 0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08, 0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b, 0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19, 0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08, 0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919, 0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08, 0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908, 0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908, 0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b, 0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908, 0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808, 0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b, 0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808, 0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808, 0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19, 0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808, 0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b, 0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808, 0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};
inline half load_f16(device const uchar *bytes) {
    ushort bits = ushort(bytes[0]) | (ushort(bytes[1]) << 8);
    return as_type<half>(bits);
}
inline uint2 gguf_k_scale_min(device const uchar *scales, uint group) {
    if (group < 4) return uint2(scales[group] & 0x3f, scales[group + 4] & 0x3f);
    return uint2(
        (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
        (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)
    );
}
inline float gguf_bf16_f32(device const uchar *bytes) {
    const uint bits = uint(bytes[0]) | (uint(bytes[1]) << 8);
    return as_type<float>(bits << 16);
}
inline float gguf_e8m0_half(uchar exponent) {
    const uint bits = exponent < 2 ? (0x00200000u << exponent) : (uint(exponent - 1) << 23);
    return as_type<float>(bits);
}
inline float gguf_mxfp4_value(uchar code) {
    const float values[8] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f };
    const float value = values[code & 7];
    return (code & 8) == 0 ? value : -value;
}
inline float gguf_weight_f32(
    device const uchar *row,
    device const ulong *iq2s_grid,
    uint tensor_type,
    uint column)
{
    if (tensor_type == 30) {
        return gguf_bf16_f32(row + ulong(column) * 2);
    }
    if (tensor_type == 39) {
        device const uchar *block = row + ulong(column >> 5) * 17;
        const uint local = column & 31;
        const uchar pair = block[1 + (local & 15)];
        const uchar code = local < 16 ? pair & 15 : pair >> 4;
        return gguf_mxfp4_value(code) * gguf_e8m0_half(block[0]);
    }
    if (tensor_type == 8) {
        device const uchar *block = row + ulong(column >> 5) * 34;
        return float(load_f16(block)) * float(as_type<char>(block[2 + (column & 31)]));
    }
    if (tensor_type == 2) {
        device const uchar *block = row + ulong(column >> 5) * 18;
        const uint local = column & 31;
        const uchar pair = block[2 + (local & 15)];
        const int quant = int(local < 16 ? pair & 15 : pair >> 4) - 8;
        return float(load_f16(block)) * float(quant);
    }

    device const uchar *block;
    const uint local = column & 255;
    if (tensor_type == 11) {
        block = row + ulong(column >> 8) * 110;
        const uint group = local >> 4;
        const uint index = local & 15;
        const uint pair = group & 7;
        const uint source = (group >> 3) * 32 + (pair & 1) * 16;
        const uint mask_source = (pair & 1) * 16;
        const uint shift = 2 * (pair >> 1);
        const uint mask = 1u << (group >> 1);
        const int low = int((block[32 + source + index] >> shift) & 3);
        const int quant = low - ((block[mask_source + index] & mask) == 0 ? 4 : 0);
        const uint scale_low = group < 8 ? block[96 + group] & 15 : block[96 + group - 8] >> 4;
        const uint scale_high = (block[104 + (group & 3)] >> (2 * (group >> 2))) & 3;
        const int scale = int(scale_low | (scale_high << 4)) - 32;
        return float(load_f16(block + 108)) * float(scale * quant);
    }
    if (tensor_type == 12 || tensor_type == 13) {
        const uint block_bytes = tensor_type == 12 ? 144 : 176;
        block = row + ulong(column >> 8) * block_bytes;
        const uint group = local >> 5;
        const uint index = local & 31;
        const uint2 scale_min = gguf_k_scale_min(block + 4, group);
        const uint low_offset = tensor_type == 12 ? 16 : 48;
        const uchar packed = block[low_offset + (group >> 1) * 32 + index];
        uint quant = (group & 1) == 0 ? packed & 15 : packed >> 4;
        if (tensor_type == 13 && (block[16 + index] & (1u << group)) != 0) quant += 16;
        return float(load_f16(block)) * float(scale_min.x * quant)
            - float(load_f16(block + 2)) * float(scale_min.y);
    }
    if (tensor_type == 14) {
        block = row + ulong(column >> 8) * 210;
        const uint half_index = local >> 7;
        const uint within = local & 127;
        const uint segment = within >> 5;
        const uint index = within & 31;
        const uint low_base = half_index * 64;
        const uint high_base = 128 + half_index * 32;
        const uint scale_base = 192 + half_index * 8 + index / 16;
        const uchar high = block[high_base + index];
        uint quant;
        int scale;
        if (segment == 0) {
            quant = (block[low_base + index] & 15) | (((high >> 0) & 3) << 4);
            scale = int(as_type<char>(block[scale_base]));
        } else if (segment == 1) {
            quant = (block[low_base + index + 32] & 15) | (((high >> 2) & 3) << 4);
            scale = int(as_type<char>(block[scale_base + 2]));
        } else if (segment == 2) {
            quant = (block[low_base + index] >> 4) | (((high >> 4) & 3) << 4);
            scale = int(as_type<char>(block[scale_base + 4]));
        } else {
            quant = (block[low_base + index + 32] >> 4) | (((high >> 6) & 3) << 4);
            scale = int(as_type<char>(block[scale_base + 6]));
        }
        return float(load_f16(block + 208)) * float(scale * (int(quant) - 32));
    }

    if (tensor_type == 18) {
        // IQ3_XXS: d(2B) + qs[0..64] 码本索引 + qs[64..96] 缩放与符号
        block = row + ulong(column >> 8) * 98;
        const uint ib32 = local >> 5;
        const uint within = local & 31;
        const uint sub = within >> 3; // 0..3
        const uint lane = within & 7; // 0..7
        device const uchar *qs = block + 2;
        const uint scales_base = 64 + ib32 * 4;
        const uint aux32 = uint(qs[scales_base]) | (uint(qs[scales_base + 1]) << 8) | (uint(qs[scales_base + 2]) << 16) | (uint(qs[scales_base + 3]) << 24);
        const float db = float(load_f16(block)) * (0.5f + float(aux32 >> 28)) * 0.5f;
        const uchar signs = ksigns_iq2xs[(aux32 >> (7 * sub)) & 127];
        const uint qs_base = ib32 * 8 + 2 * sub;
        const uint grid_index = (lane < 4) ? qs[qs_base] : qs[qs_base + 1];
        const uint grid_lane = lane & 3;
        const uchar grid_val = (iq3xxs_grid[grid_index] >> (8 * grid_lane)) & 0xff;
        const float sign = (signs & (1u << lane)) == 0 ? 1.0f : -1.0f;
        return db * float(grid_val) * sign;
    }

    if (tensor_type == 23) {
        // IQ4_XS: d(2B) + scales_h(2B) + scales_l(4B) + quants(128B) = 136B per 256 weights
        block = row + ulong(column >> 8) * 136;
        const uint ib = local >> 5; // 0..7 sub-blocks of 32
        const uint j = local & 31;  // 0..31 within sub-block
        const ushort scales_h = ushort(block[2]) | (ushort(block[3]) << 8);
        const uchar scales_l = block[4 + ib / 2];
        const uint ls = ((scales_l >> (4 * (ib & 1))) & 0x0f) | (((scales_h >> (2 * ib)) & 0x03) << 4);
        const float dl = float(load_f16(block)) * (float(ls) - 32.0f);
        const uchar q = block[8 + ib * 16 + (j & 15)];
        const uint nibble = j < 16 ? (q & 0x0f) : (q >> 4);
        return dl * float(kvalues_iq4nl[nibble]);
    }

    if (tensor_type == 20) {
        // IQ4_NL: 平铺 block,d(2B) + qs[16B] = 18B per 32 weights;
        // 字节低 nibble 是前 16 值,高 nibble 是后 16 值,无 super-block scale
        block = row + ulong(column >> 5) * 18;
        const uint j = column & 31;
        const uchar q = block[2 + (j & 15)];
        const uint nibble = j < 16 ? (q & 0x0f) : (q >> 4);
        return float(load_f16(block)) * float(kvalues_iq4nl[nibble]);
    }

    if (tensor_type == 21) {
        // IQ3_S: d(2B) + qs[64] + qh[8] + signs[32] + scales[4] = 110B per 256 weights
        block = row + ulong(column >> 8) * 110;
        const uint ib32 = local >> 5;       // 0..7
        const uint within = local & 31;
        const uint l = within >> 3;         // 0..3 (8-tuple index)
        const uint lane = within & 7;       // 0..7
        device const uchar *qs_base = block + 2;       // qs[64]
        device const uchar *qh = block + 66;            // qh[8]
        device const uchar *signs = block + 74;         // signs[32]
        device const uchar *scales = block + 106;       // scales[4]
        // 每 2 个 ib32 共用 1 个 scale 字节:ib32/2 选字节,奇偶选 nibble
        const uchar scale_byte = scales[ib32 >> 1];
        const float db = float(load_f16(block)) * (1.0f + 2.0f * float(ib32 & 1 ? scale_byte >> 4 : scale_byte & 0xf));
        // 每个 ib32 用独立 qh 字节
        const uint qh_val = uint(qh[ib32]);
        const uint qs_off = ib32 * 8 + 2 * l;
        const uint qs0 = qs_base[qs_off];
        const uint qs1 = qs_base[qs_off + 1];
        const bool is_first = lane < 4;
        const uint grid_index = is_first ? qs0 : qs1;
        const uint bit_for_qh = is_first ? (8u - 2*l) : (7u - 2*l);
        const uint grid_full = grid_index | (((qh_val << bit_for_qh) & 256u));
        const uchar grid_val = (iq3s_grid[grid_full] >> (8 * (lane & 3))) & 0xff;
        const uchar signs_byte = signs[ib32 * 4 + l];
        const float sign = (signs_byte & (1u << lane)) == 0 ? 1.0f : -1.0f;
        return db * float(grid_val) * sign;
    }

    if (tensor_type == 17) {
        // IQ2_XS: d(2B) + qs[64](32 × u16) + scales[8] = 74B per 256 weights
        block = row + ulong(column >> 8) * 74;
        const uint ib32 = local >> 5;
        const uint l = (local & 31) >> 3;   // 0..3
        const uint lane = local & 7;
        const uint q_offset = 2 + (4 * ib32 + l) * 2;
        const ushort q = ushort(block[q_offset]) | (ushort(block[q_offset + 1]) << 8);
        const uint grid_idx = q & 0x1ff;
        const uchar signs_byte = ksigns_iq2xs[q >> 9];
        const float db_base = float(load_f16(block)) * 0.25f;
        const uchar scale_byte = block[66 + ib32];
        const float db = db_base * (0.5f + float(l < 2 ? scale_byte & 0xf : scale_byte >> 4));
        const ulong grid = iq2xs_grid[grid_idx];
        const uchar grid_val = (grid >> (8 * lane)) & 0xff;
        const float sign = (signs_byte & (1u << lane)) == 0 ? 1.0f : -1.0f;
        return db * float(grid_val) * sign;
    }

    block = row + ulong(column >> 8) * 82;
    const uint group = local >> 5;
    const uint within = local & 31;
    const uint vector = within >> 3;
    const uint lane = within & 7;
    const uint high = (uint(block[66 + group]) << (8 - 2 * vector)) & 0x0300;
    const ulong grid = iq2s_grid[uint(block[2 + group * 4 + vector]) | high];
    const uint scale_bits = block[74 + group];
    const uint scale = vector < 2 ? scale_bits & 15 : scale_bits >> 4;
    const float magnitude = float((grid >> (8 * lane)) & 0xff);
    const float sign = (block[34 + group * 4 + vector] & (1u << lane)) == 0 ? 1.0f : -1.0f;
    return float(load_f16(block)) * (0.5f + float(scale)) * 0.25f * magnitude * sign;
}
inline float2 q3k_gemv2_f16(
    device const half *input,
    device const uchar *weight,
    uint columns,
    uint row_bytes,
    uint first_row,
    uint output_rows,
    uint lane)
{
    const uint block_count = columns >> 8;
    const uint tid = lane >> 2;
    const uint block_parity = lane & 3;
    const uint half_index = tid >> 2;
    const uint low_pair = 2 * ((tid & 3) >> 1);
    const uint half_row = tid & 1;
    const uint local = 8 * half_row;
    const ushort4 high_masks[4] = {
        ushort4(0x0001, 0x0100, 0x0002, 0x0200),
        ushort4(0x0004, 0x0400, 0x0008, 0x0800),
        ushort4(0x0010, 0x1000, 0x0020, 0x2000),
        ushort4(0x0040, 0x4000, 0x0080, 0x8000)
    };
    const int4 low_masks[2] = {
        int4(0x0003, 0x0300, 0x000c, 0x0c00),
        int4(0x0030, 0x3000, 0x00c0, 0xc000)
    };
    const ushort4 high_mask = high_masks[2 * half_index + low_pair / 2];
    const uint shift = 2 * low_pair;
    const float high_value1 = low_pair == 0 ? 4.0f : 64.0f;
    const float high_value2 = 4.0f * high_value1;
    const uint scale_shift1 = 4 * half_index;
    const uint scale_shift2 = scale_shift1 + low_pair;
    const uint quant_offset = 32 * half_index + local;
    const uint input_offset = 128 * half_index + 32 * low_pair + local;
    float2 sums1 = 0.0f;
    float2 sums2 = 0.0f;

    for (uint block_index = block_parity; block_index < block_count; block_index += 4) {
        float values[32];
        device const half *source = input + block_index * 256 + input_offset;
        for (uint index = 0; index < 8; ++index) {
            values[index] = float(source[index]);
            values[index + 8] = float(source[index + 16]);
            values[index + 16] = float(source[index + 32]);
            values[index + 24] = float(source[index + 48]);
        }
        #pragma unroll
        for (uint target = 0; target < 2; ++target) {
            const uint row = first_row + target;
            if (row >= output_rows) break;
            device const uchar *block = weight + ulong(row) * row_bytes + ulong(block_index) * 110;
            device const ushort *quant = (device const ushort *)(block + 32 + quant_offset);
            device const ushort *high = (device const ushort *)(block + local);
            device const ushort *scale_words = (device const ushort *)(block + 96);
            uint scales32 = uint(scale_words[4]) | (uint(scale_words[5]) << 16);
            const uint auxiliary = ((scales32 >> scale_shift2) << 4) & 0x30303030;
            scales32 = uint(scale_words[low_pair]) | (uint(scale_words[low_pair + 1]) << 16);
            scales32 = ((scales32 >> scale_shift1) & 0x0f0f0f0f) | auxiliary;
            thread const char *scales = (thread const char *)&scales32;
            float first1 = 0.0f;
            float first2 = 0.0f;
            float high1 = 0.0f;
            float second1 = 0.0f;
            float second2 = 0.0f;
            float high2 = 0.0f;
            for (uint index = 0; index < 8; index += 2) {
                const int packed = int(quant[index / 2]);
                first1 += values[index] * float(packed & low_masks[low_pair / 2][0]);
                first2 += values[index + 1] * float(packed & low_masks[low_pair / 2][1]);
                high1 += ((high[index / 2] & high_mask[0]) != 0 ? 0.0f : values[index])
                    + ((high[index / 2] & high_mask[1]) != 0 ? 0.0f : values[index + 1]);
                second1 += values[index + 16] * float(packed & low_masks[low_pair / 2][2]);
                second2 += values[index + 17] * float(packed & low_masks[low_pair / 2][3]);
                high2 += ((high[index / 2] & high_mask[2]) != 0 ? 0.0f : values[index + 16])
                    + ((high[index / 2] & high_mask[3]) != 0 ? 0.0f : values[index + 17]);
            }
            const ushort d_bits = ushort(block[108]) | (ushort(block[109]) << 8);
            const float d = float(as_type<half>(d_bits));
            const float dot1 = d * (first1 + first2 / 256.0f - high1 * high_value1);
            const float dot2 = d * (second1 + second2 / 256.0f - high2 * high_value2);
            sums1[target] += dot1 * float(scales[0] - 32);
            sums2[target] += dot2 * float(scales[2] - 32);

            first1 = first2 = high1 = second1 = second2 = high2 = 0.0f;
            for (uint index = 0; index < 8; index += 2) {
                const int packed = int(quant[index / 2 + 8]);
                first1 += values[index + 8] * float(packed & low_masks[low_pair / 2][0]);
                first2 += values[index + 9] * float(packed & low_masks[low_pair / 2][1]);
                high1 += ((high[index / 2 + 8] & high_mask[0]) != 0 ? 0.0f : values[index + 8])
                    + ((high[index / 2 + 8] & high_mask[1]) != 0 ? 0.0f : values[index + 9]);
                second1 += values[index + 24] * float(packed & low_masks[low_pair / 2][2]);
                second2 += values[index + 25] * float(packed & low_masks[low_pair / 2][3]);
                high2 += ((high[index / 2 + 8] & high_mask[2]) != 0 ? 0.0f : values[index + 24])
                    + ((high[index / 2 + 8] & high_mask[3]) != 0 ? 0.0f : values[index + 25]);
            }
            const float dot3 = d * (first1 + first2 / 256.0f - high1 * high_value1);
            const float dot4 = d * (second1 + second2 / 256.0f - high2 * high_value2);
            sums1[target] += dot3 * float(scales[1] - 32);
            sums2[target] += dot4 * float(scales[3] - 32);
        }
    }
    const float2 sums = (sums1 + 0.25f * sums2) / float(1u << shift);
    return float2(simd_sum(sums.x), simd_sum(sums.y));
}
inline float iq2s_dot32_f16(
    device const half *input,
    device const uchar *block,
    device const ulong *iq2s_grid,
    uint group)
{
    const ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
    const float d = float(as_type<half>(d_bits));
    const uint scale_bits = block[74 + group];
    float sum = 0.0f;
    for (uint vector = 0; vector < 4; ++vector) {
        const uint high = (uint(block[66 + group]) << (8 - 2 * vector)) & 0x0300;
        const ulong grid = iq2s_grid[uint(block[2 + group * 4 + vector]) | high];
        const uint scale = vector < 2 ? scale_bits & 15 : scale_bits >> 4;
        const uint signs = block[34 + group * 4 + vector];
        float vector_sum = 0.0f;
        for (uint index = 0; index < 8; ++index) {
            const float sign = (signs & (1u << index)) == 0 ? 1.0f : -1.0f;
            vector_sum += float(input[vector * 8 + index])
                * float((grid >> (8 * index)) & 0xff) * sign;
        }
        sum += d * (0.5f + float(scale)) * 0.25f * vector_sum;
    }
    return sum;
}
// block 16 字节对齐(144B/block,row_bytes 为 144 倍数):scale/min 打包区三次 uint 装载,字节提取全在寄存器。
inline uint2 q4k_scale_min(device const uchar *block, uint quant_group) {
    const uint w1 = *(device const uint *)(block + 4);
    const uint w2 = *(device const uint *)(block + 8);
    const uint w3 = *(device const uint *)(block + 12);
    uint scale;
    uint minimum;
    if (quant_group < 4) {
        scale = (w1 >> (8 * quant_group)) & 0x3f;
        minimum = (w2 >> (8 * quant_group)) & 0x3f;
    } else {
        const uint h = quant_group - 4;
        const uint b4 = (w2 >> (8 * h)) & 0xff; // block[8+h]... 注:见下原始布局
        const uint b3 = (w3 >> (8 * h)) & 0xff; // block[12+h]
        const uint b2 = (w1 >> (8 * h)) & 0xff; // block[4+h]
        scale = (b3 & 0x0f) | ((b2 >> 6) << 4);
        minimum = (b3 >> 4) | ((b4 >> 6) << 4);
    }
    return uint2(scale, minimum);
}
// IQ4_NL block(18B = 2B f16 scale + 16B nibble)与 32 元素输入的点积;
// low nibble 是前 16 元素,high nibble 是后 16(与 GGUF 布局一致)。
// qs 2B 对齐:uint16 成对读进寄存器、shift 解 nibble(替代逐字节装载);LUT 走
// threadgroup(常数内存发散索引会串行化);input 8× half4 向量读(替代 32 次标量读)。
// 三点均移植自 llama.cpp kernel_mul_mv_iq4_nl_f32。
inline float iq4nl_dot32_f16(device const half *input, device const uchar *block, threadgroup const float lut[16]) {
    const float d = float(as_type<half>(*(device const ushort *)(block)));
    device const ushort *qs16 = (device const ushort *)(block + 2);
    half4 in[8];
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        in[i] = *(device const half4 *)(input + i * 4);
    }
    float sum = 0.0f;
    #pragma unroll
    for (uint w = 0; w < 8; ++w) {
        const uint pair = qs16[w];
        const uint e = 2 * w;
        // byte 2w/2w+1 的低 nibble → 前半区,高 nibble → 后半区(+16)
        sum += float(lut[pair & 15]) * float(in[e / 4][e % 4]);
        sum += float(lut[(pair >> 4) & 15]) * float(in[4 + e / 4][e % 4]);
        sum += float(lut[(pair >> 8) & 15]) * float(in[e / 4][e % 4 + 1]);
        sum += float(lut[pair >> 12]) * float(in[4 + e / 4][e % 4 + 1]);
    }
    return d * sum;
}
inline float q4k_dot32_f16(device const half *input, device const uchar *block, uint quant_group) {
    // block 16 字节对齐(144B/block,row_bytes 为 144 倍数):全部向量化装载
    const float d = float(as_type<half>(*(device const ushort *)(block)));
    const float dmin = float(as_type<half>(*(device const ushort *)(block + 2)));
    const uint2 scale_min = q4k_scale_min(block, quant_group);
    // 32 个量化字节 = 2 x uint4;字节 i 的 nibble(奇偶组)对应 input 元素 i
    const uint4 q_lo = *(device const uint4 *)(block + 16 + (quant_group >> 1) * 32);
    const uint4 q_hi = *(device const uint4 *)(block + 16 + (quant_group >> 1) * 32 + 16);
    const uint nibble_shift = (quant_group & 1) * 4;
    float quant_sum = 0.0f;
    float input_sum = 0.0f;
    #pragma unroll
    for (uint u = 0; u < 8; ++u) {
        const uint packed = u < 4 ? q_lo[u] : q_hi[u - 4];
        const float4 in4 = float4(*(device const half4 *)(input + u * 4));
        const float4 w = float4(
            float((packed >> (0 + nibble_shift)) & 15u),
            float((packed >> (8 + nibble_shift)) & 15u),
            float((packed >> (16 + nibble_shift)) & 15u),
            float((packed >> (24 + nibble_shift)) & 15u));
        quant_sum += dot(w, in4);
        input_sum += in4.x + in4.y + in4.z + in4.w;
    }
    return d * float(scale_min.x) * quant_sum - dmin * float(scale_min.y) * input_sum;
}
// MTP verify 固定 3 个输入行：量化权重只解码一次，同时积累三行点积。
// 与单行版保持每行的 K 累加次序，减少 Q4_K 主干权重的重复读取。
inline float3 q4k_dot32x3_inputs_f16(device const half *input, uint columns, device const uchar *block, uint quant_group) {
    const float d = float(as_type<half>(*(device const ushort *)(block)));
    const float dmin = float(as_type<half>(*(device const ushort *)(block + 2)));
    const uint2 scale_min = q4k_scale_min(block, quant_group);
    const uint4 q_lo = *(device const uint4 *)(block + 16 + (quant_group >> 1) * 32);
    const uint4 q_hi = *(device const uint4 *)(block + 16 + (quant_group >> 1) * 32 + 16);
    const uint nibble_shift = (quant_group & 1) * 4;
    float3 quant_sum = 0.0f;
    float3 input_sum = 0.0f;
    #pragma unroll
    for (uint u = 0; u < 8; ++u) {
        const uint packed = u < 4 ? q_lo[u] : q_hi[u - 4];
        const float4 w = float4(
            float((packed >> (0 + nibble_shift)) & 15u),
            float((packed >> (8 + nibble_shift)) & 15u),
            float((packed >> (16 + nibble_shift)) & 15u),
            float((packed >> (24 + nibble_shift)) & 15u));
        const float4 in0 = float4(*(device const half4 *)(input + u * 4));
        const float4 in1 = float4(*(device const half4 *)(input + columns + u * 4));
        const float4 in2 = float4(*(device const half4 *)(input + 2 * columns + u * 4));
        quant_sum += float3(dot(w, in0), dot(w, in1), dot(w, in2));
        input_sum += float3(in0.x + in0.y + in0.z + in0.w, in1.x + in1.y + in1.z + in1.w, in2.x + in2.y + in2.z + in2.w);
    }
    return d * float(scale_min.x) * quant_sum - dmin * float(scale_min.y) * input_sum;
}
// 共享同一输入的两行(gate/up)融合点积:input 装载与 input_sum 只做一次,
// 两行点积按 float4 向量累积、末尾一次归约(llama 的 nr0 行共享结构的等价物)。
inline float2 q4k_dot32x2_f16(device const half *input, device const uchar *block_a, device const uchar *block_b, uint quant_group) {
    const float d_a = float(as_type<half>(*(device const ushort *)(block_a)));
    const float dmin_a = float(as_type<half>(*(device const ushort *)(block_a + 2)));
    const uint2 scale_min_a = q4k_scale_min(block_a, quant_group);
    const float d_b = float(as_type<half>(*(device const ushort *)(block_b)));
    const float dmin_b = float(as_type<half>(*(device const ushort *)(block_b + 2)));
    const uint2 scale_min_b = q4k_scale_min(block_b, quant_group);
    const uint4 qa_lo = *(device const uint4 *)(block_a + 16 + (quant_group >> 1) * 32);
    const uint4 qa_hi = *(device const uint4 *)(block_a + 16 + (quant_group >> 1) * 32 + 16);
    const uint4 qb_lo = *(device const uint4 *)(block_b + 16 + (quant_group >> 1) * 32);
    const uint4 qb_hi = *(device const uint4 *)(block_b + 16 + (quant_group >> 1) * 32 + 16);
    const uint nibble_shift = (quant_group & 1) * 4;
    float4 acc_a = 0.0f;
    float4 acc_b = 0.0f;
    float4 acc_in = 0.0f;
    #pragma unroll
    for (uint u = 0; u < 8; ++u) {
        const float4 in4 = float4(*(device const half4 *)(input + u * 4));
        acc_in += in4;
        const uint packed_a = u < 4 ? qa_lo[u] : qa_hi[u - 4];
        acc_a += float4(
            float((packed_a >> (0 + nibble_shift)) & 15u),
            float((packed_a >> (8 + nibble_shift)) & 15u),
            float((packed_a >> (16 + nibble_shift)) & 15u),
            float((packed_a >> (24 + nibble_shift)) & 15u)) * in4;
        const uint packed_b = u < 4 ? qb_lo[u] : qb_hi[u - 4];
        acc_b += float4(
            float((packed_b >> (0 + nibble_shift)) & 15u),
            float((packed_b >> (8 + nibble_shift)) & 15u),
            float((packed_b >> (16 + nibble_shift)) & 15u),
            float((packed_b >> (24 + nibble_shift)) & 15u)) * in4;
    }
    const float dot_a = acc_a.x + acc_a.y + acc_a.z + acc_a.w;
    const float dot_b = acc_b.x + acc_b.y + acc_b.z + acc_b.w;
    const float input_sum = acc_in.x + acc_in.y + acc_in.z + acc_in.w;
    return float2(
        d_a * float(scale_min_a.x) * dot_a - dmin_a * float(scale_min_a.y) * input_sum,
        d_b * float(scale_min_b.x) * dot_b - dmin_b * float(scale_min_b.y) * input_sum);
}

/// IQ3_XXS fused dot32: 对一个 ib32 组(32 权重)做 fused dequant + 点积。
/// block 指向 98 字节 block 起点;ib32 是组索引(0..7);input 指向对应 32 个 half。
inline float iq3xxs_dot32_f16(
    device const half *input,
    device const uchar *block,
    uint ib32)
{
    const float d = float(load_f16(block));
    device const uchar *qs = block + 2;
    const uint scales_base = 64 + ib32 * 4;
    const uint aux32 = uint(qs[scales_base]) | (uint(qs[scales_base + 1]) << 8) | (uint(qs[scales_base + 2]) << 16) | (uint(qs[scales_base + 3]) << 24);
    const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
    const uint qs_base = ib32 * 8;
    float sum = 0.0f;
    for (uint l = 0; l < 4; ++l) {
        const uchar signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const uint grid_idx1 = qs[qs_base + 2 * l];
        const uint grid_idx2 = qs[qs_base + 2 * l + 1];
        const uint gv1 = iq3xxs_grid[grid_idx1];
        const uint gv2 = iq3xxs_grid[grid_idx2];
        #define DOT_LANE(j) \
            (signs & (1u << (j)) ? -1.0f : 1.0f)
        sum += db * (
            DOT_LANE(0) * float((gv1 >> 0) & 0xff) * float(input[l * 8 + 0]) +
            DOT_LANE(1) * float((gv1 >> 8) & 0xff) * float(input[l * 8 + 1]) +
            DOT_LANE(2) * float((gv1 >> 16) & 0xff) * float(input[l * 8 + 2]) +
            DOT_LANE(3) * float((gv1 >> 24) & 0xff) * float(input[l * 8 + 3]) +
            DOT_LANE(4) * float((gv2 >> 0) & 0xff) * float(input[l * 8 + 4]) +
            DOT_LANE(5) * float((gv2 >> 8) & 0xff) * float(input[l * 8 + 5]) +
            DOT_LANE(6) * float((gv2 >> 16) & 0xff) * float(input[l * 8 + 6]) +
            DOT_LANE(7) * float((gv2 >> 24) & 0xff) * float(input[l * 8 + 7])
        );
        #undef DOT_LANE
    }
    return sum;
}

kernel void gguf_gemv_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &tensor_type [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= output_rows) return;
    device const half *input_row = input + ulong(group.y) * columns;
    device const uchar *weight_row = weight + ulong(group.x) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 64) {
        sum += gguf_weight_f32(weight_row, iq2s_grid, tensor_type, column) * float(input_row[column]);
    }
    threadgroup float partial[64];
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[ulong(group.y) * output_rows + group.x] = finite_f16(partial[0]);
}
kernel void gguf_gemv_accumulate_f32(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &tensor_type [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant float &route_weight [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 64) {
        sum += gguf_weight_f32(weight_row, iq2s_grid, tensor_type, column) * float(input[column]);
    }
    threadgroup float partial[64];
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[row] += float(finite_f16(partial[0])) * route_weight;
}
// 单/多行(≤8 行/threadgroup,simdgroup=输出行):权重块只解码一次,对全部
// 输入行独立累加;DSpark drafter(Q8_0)的 4 行 noise 块与 verify 直接受益。
kernel void gguf_gemv_q8_0_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &input_rows [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group.x * 8 + simd_group;
    if (row >= output_rows) return;
    const uint input_base = group.y * 8;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    const uint block_count = columns >> 5;
    const uint column = simd_lane;
    float sums[8];
    for (uint ir = 0; ir < 8; ++ir) sums[ir] = 0.0f;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        device const uchar *block = weight_row + ulong(block_index) * 34;
        const ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        const float weight_value = float(as_type<half>(d_bits)) * float(as_type<char>(block[2 + column]));
        const uint input_column = block_index * 32 + column;
        // ir 全展开让 sums 驻寄存器(动态索引会降级到 thread-local memory)
        #pragma unroll
        for (uint ir = 0; ir < 8; ++ir) {
            if (input_base + ir < input_rows) {
                sums[ir] += weight_value * float(input[ulong(input_base + ir) * columns + input_column]);
            }
        }
    }
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        const float total = simd_sum(sums[ir]);
        if (simd_lane == 0 && input_base + ir < input_rows) {
            output[ulong(input_base + ir) * output_rows + row] = finite_f16(total);
        }
    }
}
kernel void gguf_gemv_q3k_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group_position.x * 4 + simd_group * 2;
    if (first_row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    const float2 totals = q3k_gemv2_f16(input_row, weight, columns, row_bytes, first_row, output_rows, simd_lane);
    if (simd_lane == 0) {
        output[ulong(group_position.y) * output_rows + first_row] = finite_f16(totals.x);
        if (first_row + 1 < output_rows) output[ulong(group_position.y) * output_rows + first_row + 1] = finite_f16(totals.y);
    }
}
kernel void gguf_gemv_q6k_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 2 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    constexpr uchar mask1 = 0x03;
    constexpr uchar mask2 = 0x0c;
    constexpr uchar mask3 = 0x30;
    constexpr uchar mask4 = 0xc0;
    const uint lane_pair = simd_lane >> 1;
    const uint block_parity = simd_lane & 1;
    const uint half_index = lane_pair >> 3;
    const uint local = lane_pair & 7;
    const uint local4 = local * 4;
    const uint scale_offset = 8 * half_index + local4 / 16;
    const uint input_offset = 128 * half_index + local4;
    const uint low_offset = 64 * half_index + local4;
    const uint high_offset = 32 * half_index + local4;
    float sum = 0.0f;
    const uint block_count = (columns + 255) >> 8;
    for (uint block_index = block_parity; block_index < block_count; block_index += 2) {
        device const uchar *block = weight_row + ulong(block_index) * 210;
        const ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8);
        const float d = float(as_type<half>(d_bits));
        const uint column_base = block_index * 256 + input_offset;
        // 向量化装载:ql 两段与 qh 各 2 次 ushort(block 210B 仅 2B 对齐,uint 非法),
        // 输入四次 half4;寄存器内字节提取,取代 12 次标量 uchar load。
        const uint ql_first = uint(*(device const ushort *)(block + low_offset)) | (uint(*(device const ushort *)(block + low_offset + 2)) << 16);
        const uint ql_second = uint(*(device const ushort *)(block + low_offset + 32)) | (uint(*(device const ushort *)(block + low_offset + 34)) << 16);
        const uint qh4 = uint(*(device const ushort *)(block + 128 + high_offset)) | (uint(*(device const ushort *)(block + 130 + high_offset)) << 16);
        const half4 in_first = *(device const half4 *)(input_row + column_base);
        const half4 in_second = *(device const half4 *)(input_row + column_base + 32);
        const half4 in_third = *(device const half4 *)(input_row + column_base + 64);
        const half4 in_fourth = *(device const half4 *)(input_row + column_base + 96);
        float4 quant_sums = 0.0f;
        #pragma unroll
        for (uint index = 0; index < 4; ++index) {
            const uchar low_first = uchar(ql_first >> (8 * index));
            const uchar low_second = uchar(ql_second >> (8 * index));
            const uchar high = uchar(qh4 >> (8 * index));
            quant_sums.x += float(in_first[index])
                * float(int((low_first & 15) | ((high & mask1) << 4)) - 32);
            quant_sums.y += float(in_second[index])
                * float(int((low_second & 15) | ((high & mask2) << 2)) - 32);
            quant_sums.z += float(in_third[index])
                * float(int((low_first >> 4) | (high & mask3)) - 32);
            quant_sums.w += float(in_fourth[index])
                * float(int((low_second >> 4) | ((high & mask4) >> 2)) - 32);
        }
        const int4 scales = int4(
            int(as_type<char>(block[192 + scale_offset])),
            int(as_type<char>(block[194 + scale_offset])),
            int(as_type<char>(block[196 + scale_offset])),
            int(as_type<char>(block[198 + scale_offset])));
        sum += d * dot(quant_sums, float4(scales));
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) {
        output[ulong(group_position.y) * output_rows + row] = finite_f16(total);
    }
}
// gemv + 残差 epilogue(decode 单行),数值路径与 q6k gemv + add_f16 两步逐位一致。
kernel void gguf_gemv_q6k_add_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(6)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_row * 2 + simd_group;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    constexpr uchar mask1 = 0x03;
    constexpr uchar mask2 = 0x0c;
    constexpr uchar mask3 = 0x30;
    constexpr uchar mask4 = 0xc0;
    const uint lane_pair = simd_lane >> 1;
    const uint block_parity = simd_lane & 1;
    const uint half_index = lane_pair >> 3;
    const uint local = lane_pair & 7;
    const uint local4 = local * 4;
    const uint scale_offset = 8 * half_index + local4 / 16;
    const uint input_offset = 128 * half_index + local4;
    const uint low_offset = 64 * half_index + local4;
    const uint high_offset = 32 * half_index + local4;
    float sum = 0.0f;
    const uint block_count = (columns + 255) >> 8;
    for (uint block_index = block_parity; block_index < block_count; block_index += 2) {
        device const uchar *block = weight_row + ulong(block_index) * 210;
        const ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8);
        const float d = float(as_type<half>(d_bits));
        const uint column_base = block_index * 256 + input_offset;
        const uint ql_first = uint(*(device const ushort *)(block + low_offset)) | (uint(*(device const ushort *)(block + low_offset + 2)) << 16);
        const uint ql_second = uint(*(device const ushort *)(block + low_offset + 32)) | (uint(*(device const ushort *)(block + low_offset + 34)) << 16);
        const uint qh4 = uint(*(device const ushort *)(block + 128 + high_offset)) | (uint(*(device const ushort *)(block + 130 + high_offset)) << 16);
        const half4 in_first = *(device const half4 *)(input + column_base);
        const half4 in_second = *(device const half4 *)(input + column_base + 32);
        const half4 in_third = *(device const half4 *)(input + column_base + 64);
        const half4 in_fourth = *(device const half4 *)(input + column_base + 96);
        float4 quant_sums = 0.0f;
        #pragma unroll
        for (uint index = 0; index < 4; ++index) {
            const uchar low_first = uchar(ql_first >> (8 * index));
            const uchar low_second = uchar(ql_second >> (8 * index));
            const uchar high = uchar(qh4 >> (8 * index));
            quant_sums.x += float(in_first[index])
                * float(int((low_first & 15) | ((high & mask1) << 4)) - 32);
            quant_sums.y += float(in_second[index])
                * float(int((low_second & 15) | ((high & mask2) << 2)) - 32);
            quant_sums.z += float(in_third[index])
                * float(int((low_first >> 4) | (high & mask3)) - 32);
            quant_sums.w += float(in_fourth[index])
                * float(int((low_second >> 4) | ((high & mask4) >> 2)) - 32);
        }
        const int4 scales = int4(
            int(as_type<char>(block[192 + scale_offset])),
            int(as_type<char>(block[194 + scale_offset])),
            int(as_type<char>(block[196 + scale_offset])),
            int(as_type<char>(block[198 + scale_offset])));
        sum += d * dot(quant_sums, float4(scales));
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[row] = half(float(finite_f16(total)) + float(residual[row]));
}
// 设备端 embedding gather:按 token_id(设备 buffer)从 Q6_K 矩阵抠一行并 dequant 成 F16。
// 供 decode 流水线在 GPU 上闭环(argmax → embedding),CPU 无需逐 token 同步取 id。
kernel void gguf_gather_row_q6k_f16(
    device const uint *token_id [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    const uint row = token_id[0];
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    for (uint column = thread_index; column < columns; column += threads) {
        device const uchar *block = weight_row + ulong(column >> 8) * 210;
        const uint within = column & 255;
        const uint half_index = within >> 7;
        const uint segment = (within & 127) >> 5;
        const uint index = within & 31;
        const uchar low = block[64 * half_index + (segment & 1) * 32 + index];
        const uchar high = block[128 + 32 * half_index + index];
        const uint nibble = segment < 2 ? low & 15 : low >> 4;
        const int quant = int(nibble | (((high >> (2 * segment)) & 3) << 4)) - 32;
        const float scale = float(as_type<char>(block[192 + (within >> 4)]));
        const float d = float(as_type<half>(*(device const ushort *)(block + 208)));
        output[column] = finite_f16(d * scale * float(quant));
    }
}
// 设备端 embedding gather(Q4_K + embedding scale):按 token_id(设备 buffer)从 Q4_K 矩阵
// 抠一行 dequant,再乘 embedding scale,输出 F16。decode 流水线在 GPU 上闭环
// (argmax → embedding),CPU 无需逐 token 同步取 id。
// 数值逐位复刻 CPU gemma4_embedding_rows:dequant f32 → 舍入 bf16 → 乘 bf16 scale
// → 再舍入 bf16 → 转 f16(Metal 无 bfloat 类型时用位操作做 round-to-nearest-even)。
kernel void gguf_gather_row_q4k_f16(
    device const uint *token_id [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    constant float &embedding_scale [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    const uint row = token_id[0];
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    for (uint column = thread_index; column < columns; column += threads) {
        // Q4_K block:144B/256 值(d f16 + dmin f16 + scales[12] + qs[128]),
        // 布局与 CPU decode_q4_k(src/weight/codec/ggml.rs)一致。
        device const uchar *block = weight_row + ulong(column >> 8) * 144;
        const uint within = column & 255;
        const uint group = within >> 5;
        const float d = float(load_f16(block));
        const float dmin = float(load_f16(block + 2));
        const uint2 scale_min = gguf_k_scale_min(block + 4, group);
        const uchar packed = block[16 + (group >> 1) * 32 + (within & 31)];
        const uint quant = (group & 1) == 0 ? uint(packed & 0x0f) : uint(packed >> 4);
        const float value = d * float(scale_min.x) * float(quant) - dmin * float(scale_min.y);
        const float rounded = zllm_bf16_to_f32(zllm_f32_to_bf16(value));
        output[column] = half(zllm_bf16_to_f32(zllm_f32_to_bf16(rounded * embedding_scale)));
    }
}
// E4B per-layer token embedding 是 Q5_K；语义与 CPU 路径一致：dequant 后
// 先过 BF16 边界，乘 sqrt(per_layer_input_size) 后再过 BF16 边界并写 F16。
kernel void gguf_gather_row_q5k_f16(
    device const uint *token_id [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    constant float &embedding_scale [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    const uint row = token_id[0];
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    for (uint column = thread_index; column < columns; column += threads) {
        device const uchar *block = weight_row + ulong(column >> 8) * 176;
        const uint within = column & 255;
        const uint group = within >> 5;
        const uint index = within & 31;
        const uint2 scale_min = gguf_k_scale_min(block + 4, group);
        const uchar packed = block[48 + (group >> 1) * 32 + index];
        uint quant = (group & 1) == 0 ? uint(packed & 0x0f) : uint(packed >> 4);
        if ((block[16 + index] & (1u << group)) != 0) quant += 16;
        const float value = float(load_f16(block)) * float(scale_min.x * quant)
            - float(load_f16(block + 2)) * float(scale_min.y);
        const float rounded = zllm_bf16_to_f32(zllm_f32_to_bf16(value));
        output[column] = half(zllm_bf16_to_f32(zllm_f32_to_bf16(rounded * embedding_scale)));
    }
}
inline float iq4xs_row_sum_f16(
    device const half *input_row,
    device const uchar *weight_row,
    uint block_count,
    uint sub_block,
    uint within)
{
    float sum = 0.0f;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        device const uchar *block = weight_row + ulong(block_index) * 136;
        const float d = float(load_f16(block));
        const ushort scales_h = ushort(block[2]) | (ushort(block[3]) << 8);
        const uchar scales_l = block[4 + (sub_block >> 1)];
        const uint low = (sub_block & 1) != 0 ? scales_l >> 4 : scales_l & 0x0f;
        const uint ls = low | (((scales_h >> (2 * sub_block)) & 0x03) << 4);
        const float dl = d * (float(ls) - 32.0f);
        device const uchar *quants = block + 8 + sub_block * 16 + (within & 15);
        const bool high_nibble = (within & 16) != 0;
        device const half *source = input_row + block_index * 256 + sub_block * 32 + within;
        // 8 个量化字节(uint2)与 8 个 half 激活(half4 x2)各一次向量化读入,替代 16 次
        // 标量 load 的串行延迟链;nibble 提取逐字节并行。simd_shuffle 常量表方案
        // 实测更慢(7.0→4.9 tok/s)已回退,constant LUT 保持。
        const uint2 qs = *(device const uint2 *)(quants);
        const float4 src0 = float4(*(device const half4 *)(source));
        const float4 src1 = float4(*(device const half4 *)(source + 4));
        const uint2 nib = high_nibble ? ((qs >> 4) & 0x0f0f0f0fu) : (qs & 0x0f0f0f0fu);
        sum += dl * float(kvalues_iq4nl[nib.x & 15]) * src0.x;
        sum += dl * float(kvalues_iq4nl[(nib.x >> 8) & 15]) * src0.y;
        sum += dl * float(kvalues_iq4nl[(nib.x >> 16) & 15]) * src0.z;
        sum += dl * float(kvalues_iq4nl[(nib.x >> 24) & 15]) * src0.w;
        sum += dl * float(kvalues_iq4nl[nib.y & 15]) * src1.x;
        sum += dl * float(kvalues_iq4nl[(nib.y >> 8) & 15]) * src1.y;
        sum += dl * float(kvalues_iq4nl[(nib.y >> 16) & 15]) * src1.z;
        sum += dl * float(kvalues_iq4nl[(nib.y >> 24) & 15]) * src1.w;
    }
    return sum;
}

inline float iq4nl_row_sum_f16(
    device const half *input_row,
    device const uchar *weight_row,
    uint block_count,
    uint simd_lane,
    threadgroup const float lut[16])
{
    // gated gemv 用行点积:32 lane 各管一个 block(32 值),stride 32 block 扫完整行。
    // uint16 成对读 + threadgroup LUT(同 iq4nl_dot32_f16,移植 llama.cpp)。
    float sum = 0.0f;
    for (uint block_index = simd_lane; block_index < block_count; block_index += 32) {
        device const uchar *block = weight_row + ulong(block_index) * 18;
        const float d = float(load_f16(block));
        device const ushort *qs16 = (device const ushort *)(block + 2);
        device const half *source = input_row + ulong(block_index) * 32;
        #pragma unroll
        for (uint w = 0; w < 8; w += 2) {
            const uint pair0 = qs16[w];
            const uint pair1 = qs16[w + 1];
            // byte j 低 nibble → 前半区(input j),高 nibble → 后半区(input 16+j)
            const float4 v_lo = d * float4(float(lut[pair0 & 15]), float(lut[(pair0 >> 8) & 15]), float(lut[pair1 & 15]), float(lut[(pair1 >> 8) & 15]));
            const float4 v_hi = d * float4(float(lut[(pair0 >> 4) & 15]), float(lut[pair0 >> 12]), float(lut[(pair1 >> 4) & 15]), float(lut[pair1 >> 12]));
            sum += dot(v_lo, float4(*(device const half4 *)(source + 2 * w)));
            sum += dot(v_hi, float4(*(device const half4 *)(source + 16 + 2 * w)));
        }
    }
    return sum;
}

inline float iq3s_row_sum_f16(
    device const half *input_row,
    device const uchar *weight_row,
    uint block_count,
    uint ib32,
    uint tuple)
{
    float sum = 0.0f;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        device const uchar *block = weight_row + ulong(block_index) * 110;
        const float d = float(load_f16(block));
        const uchar scale_byte = block[106 + (ib32 >> 1)];
        const uint nib = (ib32 & 1) != 0 ? scale_byte >> 4 : scale_byte & 0x0f;
        const float db = d * (1.0f + 2.0f * float(nib));
        const uint qh_val = block[66 + ib32];
        device const uchar *qs = block + 2 + ib32 * 8 + tuple * 2;
        const uchar signs_byte = block[74 + ib32 * 4 + tuple];
        device const half *source = input_row + block_index * 256 + ib32 * 32 + tuple * 8;
        const uint index0 = uint(qs[0]) | ((qh_val << (8 - 2 * tuple)) & 256u);
        const uint index1 = uint(qs[1]) | ((qh_val << (7 - 2 * tuple)) & 256u);
        float acc = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            const float g0 = float((iq3s_grid[index0] >> (8 * j)) & 0xff);
            const float g1 = float((iq3s_grid[index1] >> (8 * j)) & 0xff);
            const float s0 = (signs_byte & (1u << j)) == 0 ? 1.0f : -1.0f;
            const float s1 = (signs_byte & (1u << (4 + j))) == 0 ? 1.0f : -1.0f;
            acc += g0 * s0 * float(source[j]) + g1 * s1 * float(source[4 + j]);
        }
        sum += db * acc;
    }
    return sum;
}

kernel void gguf_gemv_iq4xs_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &input_rows [[buffer(8)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    // 一个 simdgroup 32 lanes = 16 个 K 向 lane x 2 权重行(llama.cpp mul_mv_ext 的二维
    // 分割):每 lane 每 block 处理连续 16 元素。反量化(nibble→码本×scale)驻
    // 寄存器、对批内 8 个 input 行复用——行边际成本只剩 FMA,不再重复权重读与
    // LUT(MTP/DSpark verify 多行与短 prefill chunk 的收益来源;ir 循环放 block
    // 外的旧版实测 4 行 ≈ 2.4× 单行,几乎线性)。
    const uint tx = simd_lane & 15;
    const uint ty = simd_lane >> 4;
    const uint row = group_position.x * 8 + simd_group * 2 + ty;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    const uint sub_block = tx >> 1; // 16 元素段完整落在 32 元素 sub-block 内
    const bool high = (tx & 1) != 0;
    const uint ir_base = group_position.y * 8;
    const uint ir_count = min(8u, input_rows - ir_base);
    float sums[8];
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        sums[ir] = 0.0f;
    }
    for (uint block_index = 0; block_index < (columns >> 8); ++block_index) {
        device const uchar *block = weight_row + ulong(block_index) * 136;
        const float d = float(load_f16(block));
        const ushort scales_h = ushort(block[2]) | (ushort(block[3]) << 8);
        const uchar scales_l = block[4 + (sub_block >> 1)];
        const uint low = (sub_block & 1) != 0 ? scales_l >> 4 : scales_l & 0x0f;
        const uint ls = low | (((scales_h >> (2 * sub_block)) & 0x03) << 4);
        const float dl = d * (float(ls) - 32.0f);
        // GGML IQ4 布局:sub-block 的 16 字节里,前 8 字节低 nibble 是元素 0-7、
        // 后 8 字节低 nibble 是元素 8-15,两组字节的高 nibble 是元素 16-23/24-31。
        // 每 lane 处理 16 元素段:偶 tx 取低 nibble 两段,奇 tx 取高 nibble 两段。
        const uint2 qs_lo = *(device const uint2 *)(block + 8 + sub_block * 16);
        const uint2 qs_hi = *(device const uint2 *)(block + 8 + sub_block * 16 + 8);
        const uint2 nib0 = high ? ((qs_lo >> 4) & 0x0f0f0f0fu) : (qs_lo & 0x0f0f0f0fu);
        const uint2 nib1 = high ? ((qs_hi >> 4) & 0x0f0f0f0fu) : (qs_hi & 0x0f0f0f0fu);
        const float4 v0 = dl * float4(kvalues_iq4nl[nib0.x & 15], kvalues_iq4nl[(nib0.x >> 8) & 15], kvalues_iq4nl[(nib0.x >> 16) & 15], kvalues_iq4nl[(nib0.x >> 24) & 15]);
        const float4 v1 = dl * float4(kvalues_iq4nl[nib0.y & 15], kvalues_iq4nl[(nib0.y >> 8) & 15], kvalues_iq4nl[(nib0.y >> 16) & 15], kvalues_iq4nl[(nib0.y >> 24) & 15]);
        const float4 v2 = dl * float4(kvalues_iq4nl[nib1.x & 15], kvalues_iq4nl[(nib1.x >> 8) & 15], kvalues_iq4nl[(nib1.x >> 16) & 15], kvalues_iq4nl[(nib1.x >> 24) & 15]);
        const float4 v3 = dl * float4(kvalues_iq4nl[nib1.y & 15], kvalues_iq4nl[(nib1.y >> 8) & 15], kvalues_iq4nl[(nib1.y >> 16) & 15], kvalues_iq4nl[(nib1.y >> 24) & 15]);
        #pragma unroll
        for (uint ir = 0; ir < 8; ++ir) {
            if (ir < ir_count) {
                device const half *source = input + ulong(ir_base + ir) * columns + ulong(block_index) * 256 + tx * 16;
                sums[ir] += dot(v0, float4(*(device const half4 *)(source)));
                sums[ir] += dot(v1, float4(*(device const half4 *)(source + 4)));
                sums[ir] += dot(v2, float4(*(device const half4 *)(source + 8)));
                sums[ir] += dot(v3, float4(*(device const half4 *)(source + 12)));
            }
        }
    }
    // 16-lane 组内树归约:offset 都小于 16,lane0 的累加树只覆盖本行的 16 个 lane
    // (ty=1 组的中间值不进入 lane0 链)。守卫条件全 lane 一致(ir_count 为
    // dispatch 常量),simd 集体操作语义安全。
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        if (ir < ir_count) {
            float sum = sums[ir];
            sum += simd_shuffle_down(sum, 8);
            sum += simd_shuffle_down(sum, 4);
            sum += simd_shuffle_down(sum, 2);
            sum += simd_shuffle_down(sum, 1);
            if (tx == 0) {
                output[ulong(ir_base + ir) * output_rows + row] = finite_f16(sum);
            }
        }
    }
}
kernel void gguf_gemv_iq3s_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 4 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    const float sum = iq3s_row_sum_f16(input_row, weight + ulong(row) * row_bytes, columns >> 8, simd_lane >> 2, simd_lane & 3);
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[ulong(group_position.y) * output_rows + row] = finite_f16(total);
}
kernel void gguf_gated_gemv_iq4xs_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    uint group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position * 4 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input;
    const uint sub_block = simd_lane >> 2;
    const uint within = (simd_lane & 3) * 8;
    const float gate_sum = iq4xs_row_sum_f16(input_row, gate_weight + ulong(row) * gate_row_bytes, columns >> 8, sub_block, within);
    const float up_sum = iq4xs_row_sum_f16(input_row, up_weight + ulong(row) * up_row_bytes, columns >> 8, sub_block, within);
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (simd_lane == 0) {
        output[row] = finite_f16(gated_activation_value(gate_total, up_total, activation_kind, alpha, limit));
    }
}
kernel void gguf_gated_gemv_iq4nl_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    constant uint &input_rows [[buffer(14)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    // kvalues LUT 驻 threadgroup(常数内存发散索引串行化);填充必须在 early return 前
    threadgroup float iq4nl_lut[16];
    if (thread_index < 16) {
        iq4nl_lut[thread_index] = float(kvalues_iq4nl[thread_index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = group_position.x * 4 + simd_group;
    if (row >= output_rows) return;
    // grid.y = input 行:每 y 一个独立单行 gemv(单行结构 ~110GB/s;2 行共享反量化
    // 的寄存器版实测只有 ~65GB/s),权重经 L2 跨行共享。此前 kernel 只读写行 0,
    // rows=2 时输出行 1 是未初始化内存(verify 行 1 数值错的根因)。
    const uint in_row = group_position.y;
    if (in_row >= input_rows) return;
    device const half *input_row = input + ulong(in_row) * columns;
    // 顺序两段 row_sum:与 gate/up 交错单循环 A/B 实测打平(交错的双倍展开体
    // 寄存器压力抵消延迟掩盖),保留更简单的两段式
    const float gate_sum = iq4nl_row_sum_f16(input_row, gate_weight + ulong(row) * gate_row_bytes, columns >> 5, simd_lane, iq4nl_lut);
    const float up_sum = iq4nl_row_sum_f16(input_row, up_weight + ulong(row) * up_row_bytes, columns >> 5, simd_lane, iq4nl_lut);
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (simd_lane == 0) {
        output[ulong(in_row) * output_rows + row] = finite_f16(gated_activation_value(gate_total, up_total, activation_kind, alpha, limit));
    }
}
kernel void gguf_gated_gemv_iq3s_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    constant uint &input_rows [[buffer(14)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    // grid 展开为 threadgroup float4 LUT:内层一次查表替代 1 次 constant uint 读 +
    // 4 次 shift/mask/cvt(交替 A/B 实测 +13%;符号 half4 LUT 因 12KB threadgroup
    // 占压 occupancy 反而更慢,勿加)。early return 必须留在 barrier 之后。
    threadgroup float4 sg_grid[512];
    for (uint gi = thread_index; gi < 512; gi += 128) {
        const uint packed = iq3s_grid[gi];
        sg_grid[gi] = float4(float(packed & 0xffu), float((packed >> 8) & 0xffu), float((packed >> 16) & 0xffu), float((packed >> 24) & 0xffu));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 16 K-lane x 2 权重行/simdgroup;权重读一次,对每个 input token 行独立累加
    // (MTP 小批 verify / 短 prefill chunk 共享权重读带宽)。grid.y 每组承担 8 个
    // input 行,末组 ir_count 不足 8 行时截断。累加器必须是定长数组:float2 被
    // 动态索引 ir>=2 是 MSL 越界未定义行为(旧版 3..8 行静默 UB)。
    const uint tx = simd_lane & 15;
    const uint ty = simd_lane >> 4;
    const uint row = group_position.x * 8 + simd_group * 2 + ty;
    if (row >= output_rows) return;
    const uint ib32 = tx >> 1;
    const uint tuple_base = (tx & 1) * 2;
    const uint block_count = columns >> 8;
    const uint ir_base = group_position.y * 8;
    const uint ir_count = min(8u, input_rows - ir_base);
    float gate_sum[8];
    float up_sum[8];
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        gate_sum[ir] = 0.0f;
        up_sum[ir] = 0.0f;
    }
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        device const uchar *gate_block = gate_weight + ulong(row) * gate_row_bytes + ulong(block_index) * 110;
        device const uchar *up_block = up_weight + ulong(row) * up_row_bytes + ulong(block_index) * 110;
        // 110B block 只保证 2 字节对齐:d/qs/signs 用 ushort 装载取代逐字节 load
        const float gate_d = float(as_type<half>(*(device const ushort *)(gate_block)));
        const float up_d = float(as_type<half>(*(device const ushort *)(up_block)));
        const uchar gate_scale = gate_block[106 + (ib32 >> 1)];
        const uchar up_scale = up_block[106 + (ib32 >> 1)];
        const float gate_db = gate_d * (1.0f + 2.0f * float(ib32 & 1 ? gate_scale >> 4 : gate_scale & 0x0f));
        const float up_db = up_d * (1.0f + 2.0f * float(ib32 & 1 ? up_scale >> 4 : up_scale & 0x0f));
        const uint gate_qh = gate_block[66 + ib32];
        const uint up_qh = up_block[66 + ib32];
        const uint gate_qs_lo = uint(*(device const ushort *)(gate_block + 2 + ib32 * 8 + tuple_base * 2));
        const uint gate_qs_hi = uint(*(device const ushort *)(gate_block + 2 + ib32 * 8 + tuple_base * 2 + 2));
        const uint up_qs_lo = uint(*(device const ushort *)(up_block + 2 + ib32 * 8 + tuple_base * 2));
        const uint up_qs_hi = uint(*(device const ushort *)(up_block + 2 + ib32 * 8 + tuple_base * 2 + 2));
        const uint gate_signs_pair = uint(*(device const ushort *)(gate_block + 74 + ib32 * 4 + tuple_base));
        const uint up_signs_pair = uint(*(device const ushort *)(up_block + 74 + ib32 * 4 + tuple_base));
        #pragma unroll
        for (uint t = 0; t < 2; ++t) {
            const uint tuple = tuple_base + t;
            const uint gate_qs = t == 0 ? gate_qs_lo : gate_qs_hi;
            const uint up_qs = t == 0 ? up_qs_lo : up_qs_hi;
            const uint gate_signs = (gate_signs_pair >> (8 * t)) & 0xffu;
            const uint up_signs = (up_signs_pair >> (8 * t)) & 0xffu;
            const uint gate_i0 = (gate_qs & 0xffu) | ((gate_qh << (8 - 2 * tuple)) & 256u);
            const uint gate_i1 = ((gate_qs >> 8) & 0xffu) | ((gate_qh << (7 - 2 * tuple)) & 256u);
            const uint up_i0 = (up_qs & 0xffu) | ((up_qh << (8 - 2 * tuple)) & 256u);
            const uint up_i1 = ((up_qs >> 8) & 0xffu) | ((up_qh << (7 - 2 * tuple)) & 256u);
            const float4 g0 = sg_grid[gate_i0];
            const float4 g1 = sg_grid[gate_i1];
            const float4 u0 = sg_grid[up_i0];
            const float4 u1 = sg_grid[up_i1];
            const float4 gs0 = float4((gate_signs & 1) == 0 ? 1.0 : -1.0, (gate_signs & 2) == 0 ? 1.0 : -1.0, (gate_signs & 4) == 0 ? 1.0 : -1.0, (gate_signs & 8) == 0 ? 1.0 : -1.0);
            const float4 gs1 = float4((gate_signs & 16) == 0 ? 1.0 : -1.0, (gate_signs & 32) == 0 ? 1.0 : -1.0, (gate_signs & 64) == 0 ? 1.0 : -1.0, (gate_signs & 128) == 0 ? 1.0 : -1.0);
            const float4 us0 = float4((up_signs & 1) == 0 ? 1.0 : -1.0, (up_signs & 2) == 0 ? 1.0 : -1.0, (up_signs & 4) == 0 ? 1.0 : -1.0, (up_signs & 8) == 0 ? 1.0 : -1.0);
            const float4 us1 = float4((up_signs & 16) == 0 ? 1.0 : -1.0, (up_signs & 32) == 0 ? 1.0 : -1.0, (up_signs & 64) == 0 ? 1.0 : -1.0, (up_signs & 128) == 0 ? 1.0 : -1.0);
            // 符号乘积提离 ir 循环:grid 值 × 符号与行无关,不逐行重算
            const float4 gv0 = g0 * gs0;
            const float4 gv1 = g1 * gs1;
            const float4 uv0 = u0 * us0;
            const float4 uv1 = u1 * us1;
            #pragma unroll
            for (uint ir = 0; ir < 8; ++ir) {
                if (ir < ir_count) {
                    device const half *source = input + ulong(ir_base + ir) * columns + ulong(block_index) * 256 + ib32 * 32 + tuple_base * 8 + t * 8;
                    const float4 s0 = float4(*(device const half4 *)(source));
                    const float4 s1 = float4(*(device const half4 *)(source + 4));
                    gate_sum[ir] += gate_db * dot(gv0, s0) + gate_db * dot(gv1, s1);
                    up_sum[ir] += up_db * dot(uv0, s0) + up_db * dot(uv1, s1);
                }
            }
        }
    }
    // 16-lane 组内树归约(offsets < 16 不跨行):逐 input 行独立归约,守卫条件是
    // 全 lane 一致的(ir_count 为 dispatch 常量),simd 集体操作语义安全。
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        if (ir < ir_count) {
            #pragma unroll
            for (uint offset = 8; offset >= 1; offset >>= 1) {
                gate_sum[ir] += simd_shuffle_down(gate_sum[ir], offset);
                up_sum[ir] += simd_shuffle_down(up_sum[ir], offset);
            }
        }
    }
    if (tx == 0) {
        #pragma unroll
        for (uint ir = 0; ir < 8; ++ir) {
            if (ir < ir_count) {
                output[ulong(ir_base + ir) * output_rows + row] = finite_f16(gated_activation_value(gate_sum[ir], up_sum[ir], activation_kind, alpha, limit));
            }
        }
    }
}
kernel void gguf_gemv_q3k_accumulate_f32(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device float *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    constant float &route_weight [[buffer(8)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group_position.x * 4 + simd_group * 2;
    if (first_row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    const float2 totals = q3k_gemv2_f16(input_row, weight, columns, row_bytes, first_row, output_rows, simd_lane);
    if (simd_lane == 0) {
        const ulong output_index = ulong(group_position.y) * output_rows + first_row;
        output[output_index] += float(finite_f16(totals.x)) * route_weight;
        if (first_row + 1 < output_rows) output[output_index + 1] += float(finite_f16(totals.y)) * route_weight;
    }
}
kernel void gguf_gemv_iq2s_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &tensor_type [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 8 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float sum = 0.0f;
    const uint subblock_count = columns >> 5;
    for (uint subblock = simd_lane; subblock < subblock_count; subblock += 32) {
        const uint block_index = subblock >> 3;
        const uint group = subblock & 7;
        sum += iq2s_dot32_f16(
            input_row + subblock * 32,
            weight_row + ulong(block_index) * 82,
            iq2s_grid,
            group);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[ulong(group_position.y) * output_rows + row] = finite_f16(total);
}
kernel void gguf_gemv_iq3xxs_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 8 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float sum = 0.0f;
    // columns 必须是 256 的倍数;每 block 有 8 个 ib32 组,每组 32 元素
    const uint ib32_count = columns >> 5; // columns / 32
    for (uint ib32 = simd_lane; ib32 < ib32_count; ib32 += 32) {
        const uint block_index = ib32 >> 3; // 每 block 8 个 ib32
        sum += iq3xxs_dot32_f16(
            input_row + ib32 * 32,
            weight_row + ulong(block_index) * 98,
            ib32 & 7);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[ulong(group_position.y) * output_rows + row] = finite_f16(total);
}
kernel void gguf_gemv_iq2s_accumulate_f32(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    constant float &route_weight [[buffer(8)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 8 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float sum = 0.0f;
    const uint subblock_count = columns >> 5;
    for (uint subblock = simd_lane; subblock < subblock_count; subblock += 32) {
        const uint block_index = subblock >> 3;
        const uint group = subblock & 7;
        sum += iq2s_dot32_f16(
            input_row + subblock * 32,
            weight_row + ulong(block_index) * 82,
            iq2s_grid,
            group);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) {
        const ulong output_index = ulong(group_position.y) * output_rows + row;
        output[output_index] += float(finite_f16(total)) * route_weight;
    }
}
// Q/K 双路 q4k gemv:同一输入一次 dispatch 算两个矩阵(行数可不同,GQA 场景),
// 逐行结构与 gguf_gemv_q4k_f16 完全一致,省一次 dispatch 的小算子延迟。
kernel void gguf_dual_gemv_q4k_f16(
    device const half *input [[buffer(0)]],
    device const uchar *first_weight [[buffer(1)]],
    device const uchar *second_weight [[buffer(2)]],
    device half *first_output [[buffer(3)]],
    device half *second_output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &first_rows [[buffer(6)]],
    constant uint &second_rows [[buffer(7)]],
    constant uint &first_row_bytes [[buffer(8)]],
    constant uint &second_row_bytes [[buffer(9)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 8 + simd_group;
    if (row >= first_rows + second_rows) return;
    const bool is_second = row >= first_rows;
    device const uchar *weight_row = is_second
        ? second_weight + ulong(row - first_rows) * second_row_bytes
        : first_weight + ulong(row) * first_row_bytes;
    float sum = 0.0f;
    const uint group_count = columns >> 5;
    for (uint group = simd_lane; group < group_count; group += 32) {
        sum += q4k_dot32_f16(
            input + group * 32,
            weight_row + ulong(group >> 3) * 144,
            group & 7);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) {
        if (is_second) {
            second_output[row - first_rows] = finite_f16(total);
        } else {
            first_output[row] = finite_f16(total);
        }
    }
}
kernel void gguf_gemv_q4k_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 8 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float sum = 0.0f;
    const uint group_count = columns >> 5;
    for (uint group = simd_lane; group < group_count; group += 32) {
        sum += q4k_dot32_f16(
            input_row + group * 32,
            weight_row + ulong(group >> 3) * 144,
            group & 7);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[ulong(group_position.y) * output_rows + row] = finite_f16(total);
}
kernel void gguf_gemv_q4k_3m_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    uint group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position * 8 + simd_group;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float3 sum = 0.0f;
    const uint group_count = columns >> 5;
    for (uint group = simd_lane; group < group_count; group += 32) {
        sum += q4k_dot32x3_inputs_f16(
            input + group * 32,
            columns,
            weight_row + ulong(group >> 3) * 144,
            group & 7);
    }
    const float total0 = simd_sum(sum.x);
    const float total1 = simd_sum(sum.y);
    const float total2 = simd_sum(sum.z);
    if (simd_lane == 0) {
        output[row] = finite_f16(total0);
        output[output_rows + row] = finite_f16(total1);
        output[2 * output_rows + row] = finite_f16(total2);
    }
}
// gemv + 残差 epilogue(decode 单行):残差加法并入 gemv 写点,省独立 add_f16 dispatch。
// 先对 gemv 结果做 finite_f16 舍入再加残差,与两步路径(gemv → add_f16)逐位一致。
kernel void gguf_gemv_q4k_add_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(6)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_row * 8 + simd_group;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float sum = 0.0f;
    const uint group_count = columns >> 5;
    for (uint group = simd_lane; group < group_count; group += 32) {
        sum += q4k_dot32_f16(
            input + group * 32,
            weight_row + ulong(group >> 3) * 144,
            group & 7);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) output[row] = half(float(finite_f16(total)) + float(residual[row]));
}
// Q/K 双路 q4k gemv 双行变体:每 simdgroup 算 2 行(可跨 first/second 两个矩阵)。
// work item = 行 x 32 权重组;columns=1536 时 96 个 item 恰好摊满 32 lane
// (单行版 48 item,半数 lane 空闲一半时间,lane 利用率 75%→100%)。
// row0 的 lane→group 映射与单行版一致(逐位相同),row1 为再结合等价。
kernel void gguf_dual_gemv_q4k_2r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *first_weight [[buffer(1)]],
    device const uchar *second_weight [[buffer(2)]],
    device half *first_output [[buffer(3)]],
    device half *second_output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &first_rows [[buffer(6)]],
    constant uint &second_rows [[buffer(7)]],
    constant uint &first_row_bytes [[buffer(8)]],
    constant uint &second_row_bytes [[buffer(9)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row0 = group_row * 8 + simd_group * 2;
    if (row0 >= first_rows + second_rows) return;
    const bool has_second = row0 + 1 < first_rows + second_rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint group_count = columns >> 5;
    const uint work_count = group_count * (has_second ? 2 : 1);
    for (uint work = simd_lane; work < work_count; work += 32) {
        const uint row = row0 + work / group_count;
        const uint group = work % group_count;
        const bool is_second = row >= first_rows;
        device const uchar *weight_row = is_second
            ? second_weight + ulong(row - first_rows) * second_row_bytes
            : first_weight + ulong(row) * first_row_bytes;
        const float dot = q4k_dot32_f16(input + group * 32, weight_row + ulong(group >> 3) * 144, group & 7);
        if (work < group_count) { sum0 += dot; } else { sum1 += dot; }
    }
    const float total0 = simd_sum(sum0);
    const float total1 = simd_sum(sum1);
    if (simd_lane == 0) {
        if (row0 >= first_rows) { second_output[row0 - first_rows] = finite_f16(total0); } else { first_output[row0] = finite_f16(total0); }
        if (has_second) {
            const uint row1 = row0 + 1;
            if (row1 >= first_rows) { second_output[row1 - first_rows] = finite_f16(total1); } else { first_output[row1] = finite_f16(total1); }
        }
    }
}
// gemv + 残差 epilogue 双行变体:与 gguf_dual_gemv_q4k_2r_f16 同一 lane 均衡结构。
kernel void gguf_gemv_q4k_add_2r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(6)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row0 = group_row * 8 + simd_group * 2;
    if (row0 >= output_rows) return;
    const bool has_second = row0 + 1 < output_rows;
    device const uchar *weight_row0 = weight + ulong(row0) * row_bytes;
    device const uchar *weight_row1 = weight_row0 + row_bytes;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint group_count = columns >> 5;
    const uint work_count = group_count * (has_second ? 2 : 1);
    for (uint work = simd_lane; work < work_count; work += 32) {
        const uint group = work % group_count;
        device const uchar *weight_row = (work < group_count) ? weight_row0 : weight_row1;
        const float dot = q4k_dot32_f16(input + group * 32, weight_row + ulong(group >> 3) * 144, group & 7);
        if (work < group_count) { sum0 += dot; } else { sum1 += dot; }
    }
    const float total0 = simd_sum(sum0);
    const float total1 = simd_sum(sum1);
    if (simd_lane == 0) {
        output[row0] = half(float(finite_f16(total0)) + float(residual[row0]));
        if (has_second) {
            output[row0 + 1] = half(float(finite_f16(total1)) + float(residual[row0 + 1]));
        }
    }
}
kernel void gguf_gemv_qk_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &tensor_type [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 8 + simd_group;
    if (row >= output_rows) return;
    device const half *input_row = input + ulong(group_position.y) * columns;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    threadgroup uchar blocks[8 * 176];
    threadgroup uchar *block = blocks + simd_group * 176;
    const uint block_bytes = tensor_type == 12 ? 144 : 176;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    const uint block_count = (columns + 255) >> 8;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        for (uint byte = simd_lane; byte < block_bytes; byte += 32) {
            block[byte] = weight_row[ulong(block_index) * block_bytes + byte];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const ushort d_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        const ushort min_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        const float d = float(as_type<half>(d_bits));
        const float dmin = float(as_type<half>(min_bits));
        for (uint part = 0; part < 4; ++part) {
            const uint first_local = simd_lane + part * 64;
            const uint second_local = first_local + 32;
            const uint locals[2] = { first_local, second_local };
            for (uint half_index = 0; half_index < 2; ++half_index) {
                const uint local = locals[half_index];
                const uint column = block_index * 256 + local;
                const uint quant_group = local >> 5;
                const uint index = local & 31;
                uint scale;
                uint minimum;
                if (quant_group < 4) {
                    scale = block[4 + quant_group] & 0x3f;
                    minimum = block[8 + quant_group] & 0x3f;
                } else {
                    scale = (block[8 + quant_group] & 0x0f)
                        | ((block[quant_group] >> 6) << 4);
                    minimum = (block[8 + quant_group] >> 4)
                        | ((block[4 + quant_group] >> 6) << 4);
                }
                const uint low_offset = tensor_type == 12 ? 16 : 48;
                const uchar packed = block[low_offset + (quant_group >> 1) * 32 + index];
                uint quant = (quant_group & 1) == 0 ? packed & 15 : packed >> 4;
                if (tensor_type == 13 && (block[16 + index] & (1u << quant_group)) != 0) quant += 16;
                const float value = d * float(scale * quant) - dmin * float(minimum);
                if (half_index == 0) first_sum += value * float(input_row[column]);
                else second_sum += value * float(input_row[column]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float total = simd_sum(first_sum + second_sum);
    if (simd_lane == 0) output[ulong(group_position.y) * output_rows + row] = finite_f16(total);
}
kernel void gguf_gemm_rows_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &tensor_type [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &input_rows [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (group.x >= output_rows) return;
    const uint row_base = group.y * 8;
    device const uchar *weight_row = weight + ulong(group.x) * row_bytes;
    float sums[8];
    for (uint row = 0; row < 8; ++row) sums[row] = 0.0f;
    for (uint column = lane; column < columns; column += 64) {
        const float value = gguf_weight_f32(weight_row, iq2s_grid, tensor_type, column);
        for (uint row = 0; row < 8; ++row) {
            if (row_base + row < input_rows) {
                sums[row] += value * float(input[ulong(row_base + row) * columns + column]);
            }
        }
    }
    threadgroup float upper[8][32];
    if (simd_group == 1) {
        for (uint row = 0; row < 8; ++row) upper[row][simd_lane] = sums[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (uint row = 0; row < 8; ++row) {
            float sum = sums[row] + upper[row][simd_lane];
            for (uint stride = 16; stride > 0; stride >>= 1) {
                sum += simd_shuffle_down(sum, stride);
            }
            if (simd_lane == 0 && row_base + row < input_rows) {
                output[ulong(row_base + row) * output_rows + group.x] = finite_f16(sum);
            }
        }
    }
}
kernel void gguf_dequant_q3k_matrix_f16(
    device const uchar *weight [[buffer(0)]],
    device const ulong *iq2s_grid [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &output_rows [[buffer(4)]],
    constant uint &tensor_type [[buffer(5)]],
    constant uint &row_bytes [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint blocks_per_row = columns >> 8;
    if (group.x >= blocks_per_row || group.y >= output_rows || lane >= 64) return;
    device const uchar *block = weight + ulong(group.y) * row_bytes + ulong(group.x) * 110;
    const uint local = lane << 2;
    const uint quant_group = local >> 4;
    const uint index = local & 15;
    const uint pair = quant_group & 7;
    const uint source = (quant_group >> 3) * 32 + (pair & 1) * 16;
    const uint mask_source = (pair & 1) * 16;
    const uint shift = 2 * (pair >> 1);
    const uint mask = 1u << (quant_group >> 1);
    const uint scale_low = quant_group < 8 ? block[96 + quant_group] & 15 : block[96 + quant_group - 8] >> 4;
    const uint scale_high = (block[104 + (quant_group & 3)] >> (2 * (quant_group >> 2))) & 3;
    const int scale = int(scale_low | (scale_high << 4)) - 32;
    const float d = float(load_f16(block + 108));
    const ulong output_base = (ulong(group.y) * columns + ulong(group.x) * 256) + local;
    for (uint offset = 0; offset < 4; ++offset) {
        const uint low = (block[32 + source + index + offset] >> shift) & 3;
        const int quant = int(low) - ((block[mask_source + index + offset] & mask) == 0 ? 4 : 0);
        output[output_base + offset] = finite_f16(d * float(scale * quant));
    }
    (void)iq2s_grid;
    (void)tensor_type;
}
kernel void gguf_dequant_iq2s_matrix_f16(
    device const uchar *weight [[buffer(0)]],
    device const ulong *iq2s_grid [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &output_rows [[buffer(4)]],
    constant uint &tensor_type [[buffer(5)]],
    constant uint &row_bytes [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint blocks_per_row = columns >> 8;
    if (group.x >= blocks_per_row || group.y >= output_rows || lane >= 64) return;
    device const uchar *block = weight + ulong(group.y) * row_bytes + ulong(group.x) * 82;
    const uint local = lane << 2;
    const uint quant_group = local >> 5;
    const uint within = local & 31;
    const uint vector = within >> 3;
    const uint value_lane = within & 7;
    const uint high = (uint(block[66 + quant_group]) << (8 - 2 * vector)) & 0x0300;
    const ulong grid = iq2s_grid[uint(block[2 + quant_group * 4 + vector]) | high];
    const uint scale_bits = block[74 + quant_group];
    const uint scale = vector < 2 ? scale_bits & 15 : scale_bits >> 4;
    const uint signs = block[34 + quant_group * 4 + vector];
    const float d = float(load_f16(block));
    const float factor = d * (0.5f + float(scale)) * 0.25f;
    const ulong output_base = (ulong(group.y) * columns + ulong(group.x) * 256) + local;
    for (uint offset = 0; offset < 4; ++offset) {
        const uint index = value_lane + offset;
        const float magnitude = float((grid >> (8 * index)) & 0xff);
        const float sign = (signs & (1u << index)) == 0 ? 1.0f : -1.0f;
        output[output_base + offset] = finite_f16(factor * magnitude * sign);
    }
    (void)tensor_type;
}
kernel void gguf_dequant_matrix_f16(
    device const uchar *weight [[buffer(0)]],
    device const ulong *iq2s_grid [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &output_rows [[buffer(4)]],
    constant uint &tensor_type [[buffer(5)]],
    constant uint &row_bytes [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong count = ulong(output_rows) * columns;
    if (gid >= count) return;
    const uint row = gid / columns;
    const uint column = gid - row * columns;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    output[gid] = finite_f16(gguf_weight_f32(weight_row, iq2s_grid, tensor_type, column));
}
// IQ4_XS 专用 per-sub-block 反量化:generic 版每线程 1 元素,block 头(d/scales)被
// 32 个线程重复解析且单 half 写出,实测带宽仅 ~7GB/s。本版每线程负责一个 32 元素
// sub-block:头只解一次,uint2 读 16B quants,half4×8 向量化写出(prefill 大矩阵
// 反量化物化的主力路径)。
kernel void gguf_dequant_iq4xs_matrix_f16(
    device const uchar *weight [[buffer(0)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &output_rows [[buffer(4)]],
    constant uint &row_bytes [[buffer(6)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint sub_blocks_per_row = columns >> 5;
    const uint sub = group_position.x * 64 + lane;
    if (sub >= sub_blocks_per_row) return;
    const uint row = group_position.y;
    device const uchar *block = weight + ulong(row) * row_bytes + ulong(sub >> 3) * 136;
    const float d = float(load_f16(block));
    const ushort scales_h = ushort(block[2]) | (ushort(block[3]) << 8);
    const uchar scales_l = block[4 + ((sub & 7) >> 1)];
    const uint low = (sub & 1) != 0 ? scales_l >> 4 : scales_l & 0x0f;
    const uint ls = low | (((scales_h >> (2 * (sub & 7))) & 0x03) << 4);
    const float dl = d * (float(ls) - 32.0f);
    // GGML IQ4 nibble 布局:16 字节里前 8 字节低 nibble = 元素 0-7、后 8 字节低
    // nibble = 元素 8-15,两组字节的高 nibble = 元素 16-23/24-31。
    device const uchar *quants = block + 8 + (sub & 7) * 16;
    const uint2 qs_lo = *(device const uint2 *)(quants);
    const uint2 qs_hi = *(device const uint2 *)(quants + 8);
    device half *out = output + ulong(row) * columns + ulong(sub) * 32;
    #pragma unroll
    for (uint w = 0; w < 4; ++w) {
        const uint word = w == 0 ? qs_lo.x : w == 1 ? qs_lo.y : w == 2 ? qs_hi.x : qs_hi.y;
        half4 lo;
        half4 hi;
        lo.x = finite_f16(dl * float(kvalues_iq4nl[word & 15]));
        lo.y = finite_f16(dl * float(kvalues_iq4nl[(word >> 8) & 15]));
        lo.z = finite_f16(dl * float(kvalues_iq4nl[(word >> 16) & 15]));
        lo.w = finite_f16(dl * float(kvalues_iq4nl[(word >> 24) & 15]));
        hi.x = finite_f16(dl * float(kvalues_iq4nl[(word >> 4) & 15]));
        hi.y = finite_f16(dl * float(kvalues_iq4nl[(word >> 12) & 15]));
        hi.z = finite_f16(dl * float(kvalues_iq4nl[(word >> 20) & 15]));
        hi.w = finite_f16(dl * float(kvalues_iq4nl[(word >> 28) & 15]));
        *(device half4 *)(out + w * 4) = lo;
        *(device half4 *)(out + 16 + w * 4) = hi;
    }
}
// IQ3_S 专用 per-ib32 反量化(同 iq4xs 版动机):每线程一个 32 元素组,d/scale/qh
// 只读一次,grid 常量表内联,half4×4 向量化写出;覆盖 prefill 的 gate/up 大矩阵。
kernel void gguf_dequant_iq3s_matrix_f16(
    device const uchar *weight [[buffer(0)]],
    device half *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &output_rows [[buffer(4)]],
    constant uint &row_bytes [[buffer(6)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint groups_per_row = columns >> 5;
    const uint ib32 = group_position.x * 64 + lane;
    if (ib32 >= groups_per_row) return;
    const uint row = group_position.y;
    device const uchar *block = weight + ulong(row) * row_bytes + ulong(ib32 >> 3) * 110;
    const uint g = ib32 & 7;
    const float d = float(load_f16(block));
    const uchar scale = block[106 + (g >> 1)];
    const float db = d * (1.0f + 2.0f * float(g & 1 ? scale >> 4 : scale & 0x0f));
    const uint qh = block[66 + g];
    device const uchar *qs = block + 2 + g * 8;
    device const uchar *signs = block + 74 + g * 4;
    device half *out = output + ulong(row) * columns + ulong(ib32) * 32;
    #pragma unroll
    for (uint t = 0; t < 4; ++t) {
        const uint i0 = uint(qs[t * 2]) | ((qh << (8 - 2 * t)) & 256u);
        const uint i1 = uint(qs[t * 2 + 1]) | ((qh << (7 - 2 * t)) & 256u);
        const uint w0 = iq3s_grid[i0];
        const uint w1 = iq3s_grid[i1];
        const uchar sg = signs[t];
        half4 lo;
        half4 hi;
        lo.x = finite_f16(db * float(w0 & 0xff) * ((sg & 1) == 0 ? 1.0 : -1.0));
        lo.y = finite_f16(db * float((w0 >> 8) & 0xff) * ((sg & 2) == 0 ? 1.0 : -1.0));
        lo.z = finite_f16(db * float((w0 >> 16) & 0xff) * ((sg & 4) == 0 ? 1.0 : -1.0));
        lo.w = finite_f16(db * float((w0 >> 24) & 0xff) * ((sg & 8) == 0 ? 1.0 : -1.0));
        hi.x = finite_f16(db * float(w1 & 0xff) * ((sg & 16) == 0 ? 1.0 : -1.0));
        hi.y = finite_f16(db * float((w1 >> 8) & 0xff) * ((sg & 32) == 0 ? 1.0 : -1.0));
        hi.z = finite_f16(db * float((w1 >> 16) & 0xff) * ((sg & 64) == 0 ? 1.0 : -1.0));
        hi.w = finite_f16(db * float((w1 >> 24) & 0xff) * ((sg & 128) == 0 ? 1.0 : -1.0));
        *(device half4 *)(out + t * 8) = lo;
        *(device half4 *)(out + t * 8 + 4) = hi;
    }
}
// Q5_K decode GEMV 专用:每 SIMD group(32 lanes)承担一行,每 lane 连续 8 列
// (同一 32 值 quant group 内),块头 scale/min/d 只解一次,qs 连续 4 字节。
// 单/多行(≤8 行/threadgroup,simdgroup=输出行):块/scale 解码一次,全部
// 输入行独立累加;lm_head(Q5_K)的多行 verify 采样直接受益。
kernel void gguf_gemv_q5k_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &input_rows [[buffer(8)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 4 + simd_group;
    if (row >= output_rows) return;
    const uint input_base = group_position.y * 8;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    const uint block_count = columns >> 8;
    // lane 的 8 列位于 quant group = lane>>2,组内列偏移 = (lane&3)*8。
    const uint quant_group = simd_lane >> 2;
    const uint within = (simd_lane & 3) * 8;
    float sums[8];
    for (uint ir = 0; ir < 8; ++ir) sums[ir] = 0.0f;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        device const uchar *block = weight_row + ulong(block_index) * 176;
        // block 起始 8 字节对齐(row_bytes/176 都是 8 的倍数),d/dmin/quants/qh
        // 全部向量化装载,取代逐字节 load(issue-rate 是此 kernel 的主要瓶颈)。
        const float d = float(as_type<half>(*(device const ushort *)(block)));
        const float dmin = float(as_type<half>(*(device const ushort *)(block + 2)));
        const uint2 scale_min = gguf_k_scale_min(block + 4, quant_group);
        // Q5 高位 bit 在 qh[16]:列 j 的高位 = qh[j] 的 group 位。
        const uint2 quants = *(device const uint2 *)(block + 48 + (quant_group >> 1) * 32 + within);
        const uint2 qh = *(device const uint2 *)(block + 16 + within);
        // nibble 由 quant group 奇偶决定(偶组低半、奇组高半,两组共用 32 字节)。
        const bool high_nibble = (quant_group & 1) != 0;
        float weight_values[8];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint packed = (j < 4 ? quants.x : quants.y) >> (8 * (j & 3)) & 0xff;
            uint quant = high_nibble ? packed >> 4 : packed & 15;
            const uint qh_byte = (j < 4 ? qh.x : qh.y) >> (8 * (j & 3)) & 0xff;
            if ((qh_byte & (1u << quant_group)) != 0) quant += 16;
            weight_values[j] = d * float(scale_min.x * quant) - dmin * float(scale_min.y);
        }
        const float4 wv0 = float4(weight_values[0], weight_values[1], weight_values[2], weight_values[3]);
        const float4 wv1 = float4(weight_values[4], weight_values[5], weight_values[6], weight_values[7]);
        // ir 全展开让 sums 驻寄存器;input 以 half4 装载(8 标量 load 是多行
        // 放大的主因,lm_head 4 行曾 ~4x 单行)
        #pragma unroll
        for (uint ir = 0; ir < 8; ++ir) {
            if (input_base + ir < input_rows) {
                device const half *source = input + ulong(input_base + ir) * columns + block_index * 256 + quant_group * 32 + within;
                sums[ir] += dot(wv0, float4(*(device const half4 *)(source)));
                sums[ir] += dot(wv1, float4(*(device const half4 *)(source + 4)));
            }
        }
    }
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        const float total = simd_sum(sums[ir]);
        if (simd_lane == 0 && input_base + ir < input_rows) {
            output[ulong(input_base + ir) * output_rows + row] = finite_f16(total);
        }
    }
}
kernel void gguf_gemv_q5k_1r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &row_bytes [[buffer(7)]],
    constant uint &input_rows [[buffer(8)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 4 + simd_group;
    if (row >= output_rows) return;
    const uint input_base = group_position.y * 8;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    const uint block_count = columns >> 8;
    // lane 的 8 列位于 quant group = lane>>2,组内列偏移 = (lane&3)*8。
    const uint quant_group = simd_lane >> 2;
    const uint within = (simd_lane & 3) * 8;
    float sum1 = 0.0f;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        device const uchar *block = weight_row + ulong(block_index) * 176;
        // block 起始 8 字节对齐(row_bytes/176 都是 8 的倍数),d/dmin/quants/qh
        // 全部向量化装载,取代逐字节 load(issue-rate 是此 kernel 的主要瓶颈)。
        const float d = float(as_type<half>(*(device const ushort *)(block)));
        const float dmin = float(as_type<half>(*(device const ushort *)(block + 2)));
        const uint2 scale_min = gguf_k_scale_min(block + 4, quant_group);
        // Q5 高位 bit 在 qh[16]:列 j 的高位 = qh[j] 的 group 位。
        const uint2 quants = *(device const uint2 *)(block + 48 + (quant_group >> 1) * 32 + within);
        const uint2 qh = *(device const uint2 *)(block + 16 + within);
        // nibble 由 quant group 奇偶决定(偶组低半、奇组高半,两组共用 32 字节)。
        const bool high_nibble = (quant_group & 1) != 0;
        float weight_values[8];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint packed = (j < 4 ? quants.x : quants.y) >> (8 * (j & 3)) & 0xff;
            uint quant = high_nibble ? packed >> 4 : packed & 15;
            const uint qh_byte = (j < 4 ? qh.x : qh.y) >> (8 * (j & 3)) & 0xff;
            if ((qh_byte & (1u << quant_group)) != 0) quant += 16;
            weight_values[j] = d * float(scale_min.x * quant) - dmin * float(scale_min.y);
        }
        const float4 wv0 = float4(weight_values[0], weight_values[1], weight_values[2], weight_values[3]);
        const float4 wv1 = float4(weight_values[4], weight_values[5], weight_values[6], weight_values[7]);
        // ir 全展开让 sums 驻寄存器;input 以 half4 装载(8 标量 load 是多行
        // 放大的主因,lm_head 4 行曾 ~4x 单行)
        {
            device const half *source = input + ulong(input_base) * columns + block_index * 256 + quant_group * 32 + within;
            sum1 += dot(wv0, float4(*(device const half4 *)(source)));
            sum1 += dot(wv1, float4(*(device const half4 *)(source + 4)));
        }
    }
    {
        const float total = simd_sum(sum1);
        if (simd_lane == 0) {
            output[ulong(input_base) * output_rows + row] = finite_f16(total);
        }
    }
}
kernel void gguf_gated_gemv_iq3s_1r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    constant uint &input_rows [[buffer(14)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    // grid 展开为 threadgroup float4 LUT:内层一次查表替代 1 次 constant uint 读 +
    // 4 次 shift/mask/cvt(交替 A/B 实测 +13%;符号 half4 LUT 因 12KB threadgroup
    // 占压 occupancy 反而更慢,勿加)。early return 必须留在 barrier 之后。
    threadgroup float4 sg_grid[512];
    for (uint gi = thread_index; gi < 512; gi += 128) {
        const uint packed = iq3s_grid[gi];
        sg_grid[gi] = float4(float(packed & 0xffu), float((packed >> 8) & 0xffu), float((packed >> 16) & 0xffu), float((packed >> 24) & 0xffu));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 16 K-lane x 2 权重行/simdgroup;权重读一次,对每个 input token 行独立累加
    // (MTP 小批 verify / 短 prefill chunk 共享权重读带宽)。grid.y 每组承担 8 个
    // input 行,末组 ir_count 不足 8 行时截断。累加器必须是定长数组:float2 被
    // 动态索引 ir>=2 是 MSL 越界未定义行为(旧版 3..8 行静默 UB)。
    const uint tx = simd_lane & 15;
    const uint ty = simd_lane >> 4;
    const uint row = group_position.x * 8 + simd_group * 2 + ty;
    if (row >= output_rows) return;
    const uint ib32 = tx >> 1;
    const uint tuple_base = (tx & 1) * 2;
    const uint block_count = columns >> 8;
    const uint ir_base = group_position.y * 8;
    const uint ir_count = min(8u, input_rows - ir_base);
    float gate_sum1 = 0.0f;
    float up_sum1 = 0.0f;
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        device const uchar *gate_block = gate_weight + ulong(row) * gate_row_bytes + ulong(block_index) * 110;
        device const uchar *up_block = up_weight + ulong(row) * up_row_bytes + ulong(block_index) * 110;
        // 110B block 只保证 2 字节对齐:d/qs/signs 用 ushort 装载取代逐字节 load
        const float gate_d = float(as_type<half>(*(device const ushort *)(gate_block)));
        const float up_d = float(as_type<half>(*(device const ushort *)(up_block)));
        const uchar gate_scale = gate_block[106 + (ib32 >> 1)];
        const uchar up_scale = up_block[106 + (ib32 >> 1)];
        const float gate_db = gate_d * (1.0f + 2.0f * float(ib32 & 1 ? gate_scale >> 4 : gate_scale & 0x0f));
        const float up_db = up_d * (1.0f + 2.0f * float(ib32 & 1 ? up_scale >> 4 : up_scale & 0x0f));
        const uint gate_qh = gate_block[66 + ib32];
        const uint up_qh = up_block[66 + ib32];
        const uint gate_qs_lo = uint(*(device const ushort *)(gate_block + 2 + ib32 * 8 + tuple_base * 2));
        const uint gate_qs_hi = uint(*(device const ushort *)(gate_block + 2 + ib32 * 8 + tuple_base * 2 + 2));
        const uint up_qs_lo = uint(*(device const ushort *)(up_block + 2 + ib32 * 8 + tuple_base * 2));
        const uint up_qs_hi = uint(*(device const ushort *)(up_block + 2 + ib32 * 8 + tuple_base * 2 + 2));
        const uint gate_signs_pair = uint(*(device const ushort *)(gate_block + 74 + ib32 * 4 + tuple_base));
        const uint up_signs_pair = uint(*(device const ushort *)(up_block + 74 + ib32 * 4 + tuple_base));
        #pragma unroll
        for (uint t = 0; t < 2; ++t) {
            const uint tuple = tuple_base + t;
            const uint gate_qs = t == 0 ? gate_qs_lo : gate_qs_hi;
            const uint up_qs = t == 0 ? up_qs_lo : up_qs_hi;
            const uint gate_signs = (gate_signs_pair >> (8 * t)) & 0xffu;
            const uint up_signs = (up_signs_pair >> (8 * t)) & 0xffu;
            const uint gate_i0 = (gate_qs & 0xffu) | ((gate_qh << (8 - 2 * tuple)) & 256u);
            const uint gate_i1 = ((gate_qs >> 8) & 0xffu) | ((gate_qh << (7 - 2 * tuple)) & 256u);
            const uint up_i0 = (up_qs & 0xffu) | ((up_qh << (8 - 2 * tuple)) & 256u);
            const uint up_i1 = ((up_qs >> 8) & 0xffu) | ((up_qh << (7 - 2 * tuple)) & 256u);
            const float4 g0 = sg_grid[gate_i0];
            const float4 g1 = sg_grid[gate_i1];
            const float4 u0 = sg_grid[up_i0];
            const float4 u1 = sg_grid[up_i1];
            const float4 gs0 = float4((gate_signs & 1) == 0 ? 1.0 : -1.0, (gate_signs & 2) == 0 ? 1.0 : -1.0, (gate_signs & 4) == 0 ? 1.0 : -1.0, (gate_signs & 8) == 0 ? 1.0 : -1.0);
            const float4 gs1 = float4((gate_signs & 16) == 0 ? 1.0 : -1.0, (gate_signs & 32) == 0 ? 1.0 : -1.0, (gate_signs & 64) == 0 ? 1.0 : -1.0, (gate_signs & 128) == 0 ? 1.0 : -1.0);
            const float4 us0 = float4((up_signs & 1) == 0 ? 1.0 : -1.0, (up_signs & 2) == 0 ? 1.0 : -1.0, (up_signs & 4) == 0 ? 1.0 : -1.0, (up_signs & 8) == 0 ? 1.0 : -1.0);
            const float4 us1 = float4((up_signs & 16) == 0 ? 1.0 : -1.0, (up_signs & 32) == 0 ? 1.0 : -1.0, (up_signs & 64) == 0 ? 1.0 : -1.0, (up_signs & 128) == 0 ? 1.0 : -1.0);
            // 符号乘积提离 ir 循环:grid 值 × 符号与行无关,不逐行重算
            const float4 gv0 = g0 * gs0;
            const float4 gv1 = g1 * gs1;
            const float4 uv0 = u0 * us0;
            const float4 uv1 = u1 * us1;
            {
                device const half *source = input + ulong(ir_base) * columns + ulong(block_index) * 256 + ib32 * 32 + tuple_base * 8 + t * 8;
                const float4 s0 = float4(*(device const half4 *)(source));
                const float4 s1 = float4(*(device const half4 *)(source + 4));
                gate_sum1 += gate_db * dot(gv0, s0) + gate_db * dot(gv1, s1);
                up_sum1 += up_db * dot(uv0, s0) + up_db * dot(uv1, s1);
            }
        }
    }
    // 16-lane 组内树归约(offsets < 16 不跨行):逐 input 行独立归约,守卫条件是
    // 全 lane 一致的(ir_count 为 dispatch 常量),simd 集体操作语义安全。
    #pragma unroll
    for (uint offset = 8; offset >= 1; offset >>= 1) {
        gate_sum1 += simd_shuffle_down(gate_sum1, offset);
        up_sum1 += simd_shuffle_down(up_sum1, offset);
    }
    if (tx == 0) {
        output[ulong(ir_base) * output_rows + row] = finite_f16(gated_activation_value(gate_sum1, up_sum1, activation_kind, alpha, limit));
    }
}
kernel void gguf_gemv_iq4xs_1r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &input_rows [[buffer(8)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    // 一个 simdgroup 32 lanes = 16 个 K 向 lane x 2 权重行(llama.cpp mul_mv_ext 的二维
    // 分割):每 lane 每 block 处理连续 16 元素。反量化(nibble→码本×scale)驻
    // 寄存器、对批内 8 个 input 行复用——行边际成本只剩 FMA,不再重复权重读与
    // LUT(MTP/DSpark verify 多行与短 prefill chunk 的收益来源;ir 循环放 block
    // 外的旧版实测 4 行 ≈ 2.4× 单行,几乎线性)。
    const uint tx = simd_lane & 15;
    const uint ty = simd_lane >> 4;
    const uint row = group_position.x * 8 + simd_group * 2 + ty;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    const uint sub_block = tx >> 1; // 16 元素段完整落在 32 元素 sub-block 内
    const bool high = (tx & 1) != 0;
    const uint ir_base = group_position.y * 8;
    const uint ir_count = min(8u, input_rows - ir_base);
    float sum1 = 0.0f;
    for (uint block_index = 0; block_index < (columns >> 8); ++block_index) {
        device const uchar *block = weight_row + ulong(block_index) * 136;
        const float d = float(load_f16(block));
        const ushort scales_h = ushort(block[2]) | (ushort(block[3]) << 8);
        const uchar scales_l = block[4 + (sub_block >> 1)];
        const uint low = (sub_block & 1) != 0 ? scales_l >> 4 : scales_l & 0x0f;
        const uint ls = low | (((scales_h >> (2 * sub_block)) & 0x03) << 4);
        const float dl = d * (float(ls) - 32.0f);
        // GGML IQ4 布局:sub-block 的 16 字节里,前 8 字节低 nibble 是元素 0-7、
        // 后 8 字节低 nibble 是元素 8-15,两组字节的高 nibble 是元素 16-23/24-31。
        // 每 lane 处理 16 元素段:偶 tx 取低 nibble 两段,奇 tx 取高 nibble 两段。
        const uint2 qs_lo = *(device const uint2 *)(block + 8 + sub_block * 16);
        const uint2 qs_hi = *(device const uint2 *)(block + 8 + sub_block * 16 + 8);
        const uint2 nib0 = high ? ((qs_lo >> 4) & 0x0f0f0f0fu) : (qs_lo & 0x0f0f0f0fu);
        const uint2 nib1 = high ? ((qs_hi >> 4) & 0x0f0f0f0fu) : (qs_hi & 0x0f0f0f0fu);
        const float4 v0 = dl * float4(kvalues_iq4nl[nib0.x & 15], kvalues_iq4nl[(nib0.x >> 8) & 15], kvalues_iq4nl[(nib0.x >> 16) & 15], kvalues_iq4nl[(nib0.x >> 24) & 15]);
        const float4 v1 = dl * float4(kvalues_iq4nl[nib0.y & 15], kvalues_iq4nl[(nib0.y >> 8) & 15], kvalues_iq4nl[(nib0.y >> 16) & 15], kvalues_iq4nl[(nib0.y >> 24) & 15]);
        const float4 v2 = dl * float4(kvalues_iq4nl[nib1.x & 15], kvalues_iq4nl[(nib1.x >> 8) & 15], kvalues_iq4nl[(nib1.x >> 16) & 15], kvalues_iq4nl[(nib1.x >> 24) & 15]);
        const float4 v3 = dl * float4(kvalues_iq4nl[nib1.y & 15], kvalues_iq4nl[(nib1.y >> 8) & 15], kvalues_iq4nl[(nib1.y >> 16) & 15], kvalues_iq4nl[(nib1.y >> 24) & 15]);
        {
            device const half *source = input + ulong(ir_base) * columns + ulong(block_index) * 256 + tx * 16;
            sum1 += dot(v0, float4(*(device const half4 *)(source)));
            sum1 += dot(v1, float4(*(device const half4 *)(source + 4)));
            sum1 += dot(v2, float4(*(device const half4 *)(source + 8)));
            sum1 += dot(v3, float4(*(device const half4 *)(source + 12)));
        }
    }
    // 16-lane 组内树归约:offset 都小于 16,lane0 的累加树只覆盖本行的 16 个 lane
    // (ty=1 组的中间值不进入 lane0 链)。守卫条件全 lane 一致(ir_count 为
    // dispatch 常量),simd 集体操作语义安全。
    float sum = sum1;
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 1);
    if (tx == 0) {
        output[ulong(ir_base) * output_rows + row] = finite_f16(sum);
    }
}

kernel void gguf_gemv_iq4nl_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &input_rows [[buffer(8)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    // IQ4_NL 平铺 block:32 值 18 字节(d f16 + qs[16],字节低 nibble 是前 16 值)。
    // 线程组织沿用 iq4xs:一个 simdgroup = 16 个 K 向 lane x 2 权重行;每 lane
    // 每轮负责一个 block 的 16 值(偶 tx 低 nibble、奇 tx 高 nibble),16 lane 一
    // 轮覆盖 8 block = 256 值。反量化驻寄存器、对批内 8 个 input 行复用——行边际
    // 成本只剩 FMA,不再重复权重读与 LUT。
    // kvalues LUT 驻 threadgroup(常数内存发散索引串行化);填充先于 early return
    threadgroup float iq4nl_lut[16];
    if (thread_index < 16) {
        iq4nl_lut[thread_index] = float(kvalues_iq4nl[thread_index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tx = simd_lane & 15;
    const uint ty = simd_lane >> 4;
    const uint row = group_position.x * 8 + simd_group * 2 + ty;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    const bool high = (tx & 1) != 0;
    const uint ir_base = group_position.y * 8;
    const uint ir_count = min(8u, input_rows - ir_base);
    float sums[8];
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        sums[ir] = 0.0f;
    }
    const uint block_count = columns >> 5;
    for (uint block_index = tx >> 1; block_index < block_count; block_index += 8) {
        device const uchar *block = weight_row + ulong(block_index) * 18;
        const float d = float(load_f16(block));
        // qs 2B 对齐:uint16 成对读进寄存器、shift 解 nibble(移植 llama.cpp)
        device const ushort *qs16 = (device const ushort *)(block + 2);
        uint nib[16];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint pair = qs16[j];
            if (high) {
                nib[2 * j] = (pair >> 4) & 15;
                nib[2 * j + 1] = pair >> 12;
            } else {
                nib[2 * j] = pair & 15;
                nib[2 * j + 1] = (pair >> 8) & 15;
            }
        }
        const float4 v0 = d * float4(float(iq4nl_lut[nib[0]]), float(iq4nl_lut[nib[1]]), float(iq4nl_lut[nib[2]]), float(iq4nl_lut[nib[3]]));
        const float4 v1 = d * float4(float(iq4nl_lut[nib[4]]), float(iq4nl_lut[nib[5]]), float(iq4nl_lut[nib[6]]), float(iq4nl_lut[nib[7]]));
        const float4 v2 = d * float4(float(iq4nl_lut[nib[8]]), float(iq4nl_lut[nib[9]]), float(iq4nl_lut[nib[10]]), float(iq4nl_lut[nib[11]]));
        const float4 v3 = d * float4(float(iq4nl_lut[nib[12]]), float(iq4nl_lut[nib[13]]), float(iq4nl_lut[nib[14]]), float(iq4nl_lut[nib[15]]));
        #pragma unroll
        for (uint ir = 0; ir < 8; ++ir) {
            if (ir < ir_count) {
                device const half *source = input + ulong(ir_base + ir) * columns + ulong(block_index) * 32 + (tx & 1) * 16;
                sums[ir] += dot(v0, float4(*(device const half4 *)(source)));
                sums[ir] += dot(v1, float4(*(device const half4 *)(source + 4)));
                sums[ir] += dot(v2, float4(*(device const half4 *)(source + 8)));
                sums[ir] += dot(v3, float4(*(device const half4 *)(source + 12)));
            }
        }
    }
    // 16-lane 组内树归约:offset 都小于 16,lane0 的累加树只覆盖本行的 16 个 lane
    // (ty=1 组的中间值不进入 lane0 链)。守卫条件全 lane 一致(ir_count 为
    // dispatch 常量),simd 集体操作语义安全。
    #pragma unroll
    for (uint ir = 0; ir < 8; ++ir) {
        if (ir < ir_count) {
            float sum = sums[ir];
            sum += simd_shuffle_down(sum, 8);
            sum += simd_shuffle_down(sum, 4);
            sum += simd_shuffle_down(sum, 2);
            sum += simd_shuffle_down(sum, 1);
            if (tx == 0) {
                output[ulong(ir_base + ir) * output_rows + row] = finite_f16(sum);
            }
        }
    }
}


// 单行(decode)专用变体:去掉 8 行累加器的寄存器压力,结构同 gguf_gemv_iq4xs_1r_f16
// verify(4 行)专属 IQ4_NL gemv:与 gguf_gemv_iq4nl_1r 同构(每 simdgroup 一行
// 权重、32 K-lane 全宽),唯一差异是每 block 反量化一次后对 4 个 input 行各做
// dot——权重读/LUT 带宽与单行 decode 版相同(实测 115GB/s),行边际只剩 FMA。
kernel void gguf_gemv_iq4nl_4r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &input_rows [[buffer(8)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_position.x * 4 + simd_group;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    float sums[4];
    #pragma unroll
    for (uint ir = 0; ir < 4; ++ir) sums[ir] = 0.0f;
    const uint block_count = columns >> 5;
    float w[32];
    for (uint block_index = simd_lane; block_index < block_count; block_index += 32) {
        // 反量化一次进寄存器(LUT/读只做一遍),4 行只付 FMA
        device const uchar *block = weight_row + ulong(block_index) * 18;
        const float d = float(as_type<half>(*(device const ushort *)(block)));
        device const uchar *qs = block + 2;
        // IQ4_NL nibble 布局:字节 j 低 nibble = 值 j,高 nibble = 值 16+j
        #pragma unroll
        for (uint u = 0; u < 16; ++u) {
            const uchar byte = qs[u];
            w[u] = d * float(kvalues_iq4nl[byte & 15]);
            w[16 + u] = d * float(kvalues_iq4nl[byte >> 4]);
        }
        #pragma unroll
        for (uint ir = 0; ir < 4; ++ir) {
            if (ir < input_rows) {
                device const half *input_row = input + ulong(ir) * columns + ulong(block_index) * 32;
                float sum = 0.0f;
                #pragma unroll
                for (uint u = 0; u < 32; ++u) {
                    sum += w[u] * float(input_row[u]);
                }
                sums[ir] += sum;
            }
        }
    }
    #pragma unroll
    for (uint ir = 0; ir < 4; ++ir) {
        if (ir < input_rows) {
            const float total = simd_sum(sums[ir]);
            if (simd_lane == 0) {
                output[ulong(ir) * output_rows + row] = finite_f16(total);
            }
        }
    }
}

kernel void gguf_gemv_iq4nl_1r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device half *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &output_rows [[buffer(5)]],
    constant uint &input_rows [[buffer(8)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group_position [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup float iq4nl_lut[16];
    if (thread_index < 16) {
        iq4nl_lut[thread_index] = float(kvalues_iq4nl[thread_index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 每 simdgroup 一行(32 lane 全用,block 交错步进 32),与 gguf_gemv_q4k_f16
    // 同构;原 2 行/simdgroup 布局每行仅 16 lane,利用率减半。
    const uint row = group_position.x * 4 + simd_group;
    if (row >= output_rows) return;
    device const uchar *weight_row = weight + ulong(row) * row_bytes;
    const uint ir_base = group_position.y * 8;
    device const half *input_row = input + ulong(ir_base) * columns;
    float sum = 0.0f;
    const uint block_count = columns >> 5;
    for (uint block_index = simd_lane; block_index < block_count; block_index += 32) {
        sum += iq4nl_dot32_f16(input_row + ulong(block_index) * 32, weight_row + ulong(block_index) * 18, iq4nl_lut);
    }
    const float total = simd_sum(sum);
    if (simd_lane == 0) {
        output[ulong(ir_base) * output_rows + row] = finite_f16(total);
    }
}
// 2-simdgroup / TG 版本的 IQ4_XS gemv:N_SG=2 × N_R0=1,每 simdgroup 用 32 lane
// 算 1 完整行 K 维(每 lane 8 element,8 sub-block × 4 lane/sub-block 跨 simdgroup)。
// 跨 simdgroup 写不同 row,无需 shared memory 同步——单请求 decode 利用率
// 从单 simdgroup 翻倍。设计对应 llama.cpp master N_SG_IQ4_XS=2 / N_R0_IQ4_XS=1。
kernel void gguf_gated_gemv_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device const ulong *iq2s_grid [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_type [[buffer(7)]],
    constant uint &up_type [[buffer(8)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    if (row >= output_rows) return;
    device const uchar *gate_row = gate_weight + ulong(row) * gate_row_bytes;
    device const uchar *up_row = up_weight + ulong(row) * up_row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint column = lane; column < columns; column += 64) {
        const float input_value = float(input[column]);
        gate_sum += gguf_weight_f32(gate_row, iq2s_grid, gate_type, column) * input_value;
        up_sum += gguf_weight_f32(up_row, iq2s_grid, up_type, column) * input_value;
    }
    threadgroup float gate_partial[64];
    threadgroup float up_partial[64];
    gate_partial[lane] = gate_sum;
    up_partial[lane] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            gate_partial[lane] += gate_partial[lane + stride];
            up_partial[lane] += up_partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const half gate_f16 = finite_f16(gate_partial[0]);
        const half up_f16 = finite_f16(up_partial[0]);
        output[row] = finite_f16(gated_activation_value(
            float(gate_f16), float(up_f16), activation_kind, alpha, limit));
    }
}
kernel void gguf_gated_gemv_q4k_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device const ulong *iq2s_grid [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_type [[buffer(7)]],
    constant uint &up_type [[buffer(8)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    (void)iq2s_grid;
    (void)gate_type;
    (void)up_type;
    const uint row = group_row * 8 + simd_group;
    if (row >= output_rows) return;
    device const uchar *gate_row = gate_weight + ulong(row) * gate_row_bytes;
    device const uchar *up_row = up_weight + ulong(row) * up_row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint group_count = columns >> 5;
    for (uint group = simd_lane; group < group_count; group += 32) {
        const float2 pair = q4k_dot32x2_f16(input + group * 32, gate_row + ulong(group >> 3) * 144, up_row + ulong(group >> 3) * 144, group & 7);
        gate_sum += pair.x;
        up_sum += pair.y;
    }
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (simd_lane == 0) {
        const half gate_f16 = finite_f16(gate_total);
        const half up_f16 = finite_f16(up_total);
        output[row] = finite_f16(gated_activation_value(
            float(gate_f16), float(up_f16), activation_kind, alpha, limit));
    }
}
// 双行变体:每个 simdgroup 算 2 个输出行的 gate/up,work item = 行 × 32 权重组。
// columns=1536 时 96 个 item 恰好摊满 32  lane(1 行只有 48 个,半数 lane 空闲一半时间);
// input 在 gate/up 间共享(见 q4k_dot32x2_f16),行间不共享但 lane 利用率 75%→100%。
kernel void gguf_gated_gemv_q4k_2r_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device const ulong *iq2s_grid [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_type [[buffer(7)]],
    constant uint &up_type [[buffer(8)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    (void)iq2s_grid;
    (void)gate_type;
    (void)up_type;
    const uint first_row = group_row * 8 + simd_group * 2;
    if (first_row >= output_rows) return;
    device const uchar *gate_row0 = gate_weight + ulong(first_row) * gate_row_bytes;
    device const uchar *up_row0 = up_weight + ulong(first_row) * up_row_bytes;
    const bool has_second = first_row + 1 < output_rows;
    device const uchar *gate_row1 = gate_row0 + gate_row_bytes;
    device const uchar *up_row1 = up_row0 + up_row_bytes;
    float gate_sum0 = 0.0f;
    float up_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum1 = 0.0f;
    const uint group_count = columns >> 5;
    const uint work_count = group_count * (has_second ? 2 : 1);
    for (uint work = simd_lane; work < work_count; work += 32) {
        const uint group = work % group_count;
        device const half *input_group = input + group * 32;
        if (work < group_count) {
            const float2 pair = q4k_dot32x2_f16(input_group, gate_row0 + ulong(group >> 3) * 144, up_row0 + ulong(group >> 3) * 144, group & 7);
            gate_sum0 += pair.x;
            up_sum0 += pair.y;
        } else {
            const float2 pair = q4k_dot32x2_f16(input_group, gate_row1 + ulong(group >> 3) * 144, up_row1 + ulong(group >> 3) * 144, group & 7);
            gate_sum1 += pair.x;
            up_sum1 += pair.y;
        }
    }
    const float gate_total0 = simd_sum(gate_sum0);
    const float up_total0 = simd_sum(up_sum0);
    const float gate_total1 = simd_sum(gate_sum1);
    const float up_total1 = simd_sum(up_sum1);
    if (simd_lane == 0) {
        output[first_row] = finite_f16(gated_activation_value(
            float(finite_f16(gate_total0)), float(finite_f16(up_total0)), activation_kind, alpha, limit));
        if (has_second) {
            output[first_row + 1] = finite_f16(gated_activation_value(
                float(finite_f16(gate_total1)), float(finite_f16(up_total1)), activation_kind, alpha, limit));
        }
    }
}
kernel void gguf_gated_gemv_q3k_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device const ulong *iq2s_grid [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_type [[buffer(7)]],
    constant uint &up_type [[buffer(8)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint first_row = group_row * 4 + simd_group * 2;
    if (first_row >= output_rows) return;
    const float2 gate_totals = q3k_gemv2_f16(input, gate_weight, columns, gate_row_bytes, first_row, output_rows, simd_lane);
    const float2 up_totals = q3k_gemv2_f16(input, up_weight, columns, up_row_bytes, first_row, output_rows, simd_lane);
    if (simd_lane == 0) {
        const half gate_first = finite_f16(gate_totals.x);
        const half up_first = finite_f16(up_totals.x);
        output[first_row] = finite_f16(gated_activation_value(
            float(gate_first), float(up_first), activation_kind, alpha, limit));
        if (first_row + 1 < output_rows) {
            const half gate_second = finite_f16(gate_totals.y);
            const half up_second = finite_f16(up_totals.y);
            output[first_row + 1] = finite_f16(gated_activation_value(
                float(gate_second), float(up_second), activation_kind, alpha, limit));
        }
    }
}
kernel void gguf_gated_gemv_iq2s_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device const ulong *iq2s_grid [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_type [[buffer(7)]],
    constant uint &up_type [[buffer(8)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_row * 8 + simd_group;
    if (row >= output_rows) return;
    device const uchar *gate_row = gate_weight + ulong(row) * gate_row_bytes;
    device const uchar *up_row = up_weight + ulong(row) * up_row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint subblock_count = columns >> 5;
    for (uint subblock = simd_lane; subblock < subblock_count; subblock += 32) {
        const uint block_index = subblock >> 3;
        const uint group = subblock & 7;
        device const half *input_block = input + subblock * 32;
        gate_sum += iq2s_dot32_f16(input_block, gate_row + ulong(block_index) * 82, iq2s_grid, group);
        up_sum += iq2s_dot32_f16(input_block, up_row + ulong(block_index) * 82, iq2s_grid, group);
    }
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (simd_lane == 0) {
        const half gate_f16 = finite_f16(gate_total);
        const half up_f16 = finite_f16(up_total);
        output[row] = finite_f16(gated_activation_value(
            float(gate_f16), float(up_f16), activation_kind, alpha, limit));
    }
}
kernel void gguf_gated_gemv_iq3xxs_f16(
    device const half *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device const ulong *iq2s_grid [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &output_rows [[buffer(6)]],
    constant uint &gate_row_bytes [[buffer(9)]],
    constant uint &up_row_bytes [[buffer(10)]],
    constant uint &activation_kind [[buffer(11)]],
    constant float &alpha [[buffer(12)]],
    constant float &limit [[buffer(13)]],
    uint group_row [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint row = group_row * 8 + simd_group;
    if (row >= output_rows) return;
    device const uchar *gate_row = gate_weight + ulong(row) * gate_row_bytes;
    device const uchar *up_row = up_weight + ulong(row) * up_row_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    const uint ib32_count = columns >> 5;
    for (uint ib32 = simd_lane; ib32 < ib32_count; ib32 += 32) {
        const uint block_index = ib32 >> 3;
        device const half *input_block = input + ib32 * 32;
        gate_sum += iq3xxs_dot32_f16(input_block, gate_row + ulong(block_index) * 98, ib32 & 7);
        up_sum += iq3xxs_dot32_f16(input_block, up_row + ulong(block_index) * 98, ib32 & 7);
    }
    const float gate_total = simd_sum(gate_sum);
    const float up_total = simd_sum(up_sum);
    if (simd_lane == 0) {
        output[row] = finite_f16(gated_activation_value(gate_total, up_total, activation_kind, alpha, limit));
    }
}
kernel void prefetch_shared_pages(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant ulong &length [[buffer(2)]],
    uint page [[thread_position_in_grid]])
{
    const ulong offset = ulong(page) * 16384;
    if (offset < length) output[page] = input[offset];
}
// Fused Q3_K GEMM (prefill 用: M >= 4 input rows × N output cols)
// 每个 thread group = 4 simdgroup × 32 thread = 128 thread
// 4 simdgroup 算 4 input rows × 4 output cols = 16 outputs per group
// dispatch: (N / 4, M / 4)
kernel void gguf_gemm_q3k_fused_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint in_row = gid.y * 4 + simd_gid;
    if (in_row >= M) return;
    device const half *input_row = input + ulong(in_row) * K;
    const uint first_col = gid.x * 4;
    // q3k_gemv2_f16 returns float2: (first_row, first_row+1) 的 dot
    // 调两次算 4 个 output cols
    const float2 r1 = q3k_gemv2_f16(input_row, weight, K, row_bytes, first_col, N, simd_lane);
    const float2 r2 = q3k_gemv2_f16(input_row, weight, K, row_bytes, first_col + 2, N, simd_lane);
    if (simd_lane == 0) {
        if (first_col < N)     output[ulong(in_row) * N + first_col]     = finite_f16(r1.x);
        if (first_col + 1 < N) output[ulong(in_row) * N + first_col + 1] = finite_f16(r1.y);
        if (first_col + 2 < N) output[ulong(in_row) * N + first_col + 2] = finite_f16(r2.x);
        if (first_col + 3 < N) output[ulong(in_row) * N + first_col + 3] = finite_f16(r2.y);
    }
}
// Fused Q6_K GEMM (prefill 用: M >= 4 input rows × N output cols)
// 每个 thread group = 4 simdgroup × 32 thread = 128 thread
// 4 simdgroup 算 4 input rows × 4 output cols = 16 outputs per group
// dispatch: (N / 4, M / 4)
kernel void gguf_gemm_q6k_fused_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    const uint in_row = gid.y * 4 + simd_gid;
    if (in_row >= M) return;
    device const half *input_row = input + ulong(in_row) * K;
    const uint first_col = gid.x * 4;
    constexpr uchar mask1 = 0x03;
    constexpr uchar mask2 = 0x0c;
    constexpr uchar mask3 = 0x30;
    constexpr uchar mask4 = 0xc0;
    const uint lane_pair = simd_lane >> 1;
    const uint block_parity = simd_lane & 1;
    const uint half_index = lane_pair >> 3;
    const uint local = lane_pair & 7;
    const uint local4 = local * 4;
    const uint scale_offset = 8 * half_index + local4 / 16;
    const uint input_offset = 128 * half_index + local4;
    const uint low_offset = 64 * half_index + local4;
    const uint high_offset = 32 * half_index + local4;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint col_off = 0; col_off < 4; ++col_off) {
        const uint out_col = first_col + col_off;
        if (out_col >= N) break;
        device const uchar *weight_row = weight + ulong(out_col) * row_bytes;
        const uint block_count = (K + 255) >> 8;
        for (uint block_index = block_parity; block_index < block_count; block_index += 2) {
            device const uchar *block = weight_row + ulong(block_index) * 210;
            const ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8);
            const float d = float(as_type<half>(d_bits));
            const uint column_base = block_index * 256 + input_offset;
            float4 quant_sums = 0.0f;
            for (uint index = 0; index < 4; ++index) {
                const uchar low_first = block[low_offset + index];
                const uchar low_second = block[low_offset + 32 + index];
                const uchar high = block[128 + high_offset + index];
                quant_sums.x += float(input_row[column_base + index])
                    * float(int((low_first & 15) | ((high & mask1) << 4)) - 32);
                quant_sums.y += float(input_row[column_base + 32 + index])
                    * float(int((low_second & 15) | ((high & mask2) << 2)) - 32);
                quant_sums.z += float(input_row[column_base + 64 + index])
                    * float(int((low_first >> 4) | (high & mask3)) - 32);
                quant_sums.w += float(input_row[column_base + 96 + index])
                    * float(int((low_second >> 4) | ((high & mask4) >> 2)) - 32);
            }
            const int4 scales = int4(
                int(as_type<char>(block[192 + scale_offset])),
                int(as_type<char>(block[194 + scale_offset])),
                int(as_type<char>(block[196 + scale_offset])),
                int(as_type<char>(block[198 + scale_offset])));
            sums[col_off] += d * dot(quant_sums, float4(scales));
        }
    }
    // 4 simdgroup reduce, 每个 simdgroup 算自己的 (in_row, first_col..first_col+3)
    sums[0] = simd_sum(sums[0]);
    sums[1] = simd_sum(sums[1]);
    sums[2] = simd_sum(sums[2]);
    sums[3] = simd_sum(sums[3]);
    if (simd_lane == 0) {
        if (first_col < N)     output[ulong(in_row) * N + first_col]     = finite_f16(sums[0]);
        if (first_col + 1 < N) output[ulong(in_row) * N + first_col + 1] = finite_f16(sums[1]);
        if (first_col + 2 < N) output[ulong(in_row) * N + first_col + 2] = finite_f16(sums[2]);
        if (first_col + 3 < N) output[ulong(in_row) * N + first_col + 3] = finite_f16(sums[3]);
    }
}

// Fused IQ4_NL GEMM：64×64 output tile，每个 K=32 tile 只反量化一次权重到
// threadgroup，4 个 simdgroup 用 8×8 MMA 复用。不能按 input row 扩 GEMV：那会
// 让长 prompt 为每行重读整份权重，虽数值正确却比 dequant+MPS 更慢。
#if __METAL_VERSION__ >= 400
// M5 Metal 4 cooperative tensor 路径。逻辑矩阵为 weight[N,K] × input[K,M]，
// 物理输出仍是 zLLM 行优先 output[M,N]。每组覆盖 64 个权重行 × 128 个
// token；量化权重 tile 反量化进 threadgroup，input 直接从 device tensor 读取。
kernel void gguf_gemm_iq4nl_mpp_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &input_rows [[buffer(4)]],
    constant uint &weight_rows [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    constexpr int TILE_WEIGHT_ROWS = 64;
    constexpr int TILE_INPUT_ROWS = 128;
    constexpr int TILE_K = 32;
    constexpr int SIMD_GROUPS = 4;
    threadgroup half stage_weight[TILE_WEIGHT_ROWS * TILE_K];
    const int weight_base = int(group.y) * TILE_WEIGHT_ROWS;
    const int input_base = int(group.x) * TILE_INPUT_ROWS;

    auto staged = tensor(stage_weight, dextents<int32_t, 2>(TILE_K, TILE_WEIGHT_ROWS));
    device half *input_mut = const_cast<device half *>(input);
    auto input_tensor = tensor(input_mut, dextents<int32_t, 2>(int(K), int(input_rows)), array<int, 2>({1, int(K)}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            TILE_INPUT_ROWS,
            TILE_WEIGHT_ROWS,
            static_cast<int>(dynamic_extent),
            false,
            true,
            true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<SIMD_GROUPS>> multiply;
    auto accumulator = multiply.get_destination_cooperative_tensor<decltype(input_tensor), decltype(staged), float>();

    for (int k_base = 0; k_base < int(K); k_base += TILE_K) {
        // 128 threads：每线程负责一个权重行的半个 IQ4_NL block（16 值）。
        const int local_weight_row = int(thread_index) >> 1;
        const int half_block = int(thread_index) & 1;
        const int out_row = weight_base + local_weight_row;
        const int local_k = half_block * 16;
        if (out_row < int(weight_rows)) {
            device const uchar *block = weight + ulong(out_row) * row_bytes + ulong(k_base >> 5) * 18;
            const float scale = float(as_type<half>(*(device const ushort *)block));
            for (int i = 0; i < 16; ++i) {
                const uchar packed = block[2 + i];
                const uint code = half_block == 0 ? packed & 15 : packed >> 4;
                stage_weight[local_weight_row * TILE_K + local_k + i] = half(scale * float(kvalues_iq4nl[code]));
            }
        } else {
            for (int i = 0; i < 16; ++i) {
                stage_weight[local_weight_row * TILE_K + local_k + i] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int k_extent = min(TILE_K, int(K) - k_base);
        auto weight_tile = tensor(stage_weight, dextents<int32_t, 2>(k_extent, TILE_WEIGHT_ROWS), array<int, 2>({1, TILE_K}));
        auto input_tile = tensor(input_mut + k_base + ulong(input_base) * K, dextents<int32_t, 2>(k_extent, int(input_rows) - input_base), array<int, 2>({1, int(K)}));
        multiply.run(input_tile, weight_tile, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto destination = tensor(output, dextents<int32_t, 2>(int(weight_rows), int(input_rows)), array<int, 2>({1, int(weight_rows)}));
    accumulator.store(destination.slice(weight_base, input_base));
}
#endif

kernel void gguf_gemm_iq4nl_fused_f16(
    device const half *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const ulong *iq2s_grid [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &M [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &K [[buffer(6)]],
    constant uint &row_bytes [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup half stage_a[64 * 32];
    threadgroup half stage_b[32 * 64];
    threadgroup float result[64 * 64];
    const uint row_base = group.y * 64;
    const uint col_base = group.x * 64;
    const uint gi = simd_index & 1;
    const uint gj = simd_index >> 1;
    simdgroup_float8x8 acc[4][4];
    for (uint mi = 0; mi < 4; ++mi) {
        for (uint nj = 0; nj < 4; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k_base = 0; k_base < K; k_base += 32) {
        for (uint index = thread_index; index < (64 * 32) / 4; index += 128) {
            const uint local_row = (index * 4) >> 5;
            const uint local_k = (index * 4) & 31;
            half4 value = 0.0h;
            if (row_base + local_row < M) {
                value = *(device const half4 *)(input + ulong(row_base + local_row) * K + k_base + local_k);
            }
            *(threadgroup half4 *)(stage_a + local_row * 32 + local_k) = value;
        }
        for (uint index = thread_index; index < (32 * 64) / 4; index += 128) {
            const uint flat = index * 4;
            #pragma unroll
            for (uint item = 0; item < 4; ++item) {
                const uint element = flat + item;
                const uint local_k = element >> 6;
                const uint local_n = element & 63;
                const uint out_col = col_base + local_n;
                half value = 0.0h;
                if (out_col < N) {
                    const uint k = k_base + local_k;
                    device const uchar *block = weight + ulong(out_col) * row_bytes + ulong(k >> 5) * 18;
                    const float d = float(as_type<half>(*(device const ushort *)block));
                    const uchar packed = block[2 + (k & 15)];
                    const uint code = (k & 16) == 0 ? packed & 15 : packed >> 4;
                    value = half(d * float(kvalues_iq4nl[code]));
                }
                stage_b[local_k * 64 + local_n] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint mi = 0; mi < 4; ++mi) {
                simdgroup_load(a[mi], stage_a + (gi * 32 + mi * 8) * 32 + k, 32);
            }
            for (uint nj = 0; nj < 4; ++nj) {
                simdgroup_load(b[nj], stage_b + k * 64 + gj * 32 + nj * 8, 64);
            }
            for (uint mi = 0; mi < 4; ++mi) {
                for (uint nj = 0; nj < 4; ++nj) {
                    simdgroup_multiply_accumulate(acc[mi][nj], a[mi], b[nj], acc[mi][nj]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mi = 0; mi < 4; ++mi) {
        for (uint nj = 0; nj < 4; ++nj) {
            simdgroup_store(acc[mi][nj], result + (gi * 32 + mi * 8) * 64 + gj * 32 + nj * 8, 64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = thread_index; index < 64 * 64; index += 128) {
        const uint local_row = index >> 6;
        const uint local_col = index & 63;
        if (row_base + local_row < M && col_base + local_col < N) {
            output[ulong(row_base + local_row) * N + col_base + local_col] = finite_f16(result[index]);
        }
    }
}
"#;

use crate::backend::metal::api as metal;

use super::dense::{GatedActivation, launch_matmul_f16};
use super::moe::MetalF32Accumulator;
use super::tensor::gated_activation_tensor;
use super::{Activation, MTLSize, MetalContext, MetalTensor, MetalTensorDType, as_bytes, launch_1d, mem, set_bytes, to_f16_tensor, validate_size, validate_u32};

use std::sync::OnceLock;

#[cfg(test)]
use crate::backend::metal::MetalWeight;
#[cfg(test)]
use half::f16;

/// 聚合基础与 MoE 子模块 shader,供 `super::kernels_source()` 拼接。
/// kernel 之间互不调用,共享 helper 都在文件内先于使用点或全局 preamble。
pub fn shaders() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| [BASE_SHADERS, experts::SHADERS].concat()).as_str()
}
