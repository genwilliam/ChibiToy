from __future__ import annotations

# 正向提示词，包含固定的构图、材质、光照、摄影风格等描述
POSITIVE_PROMPT = (

    "chibi toy figure, collectible designer toy, 1.75 head-to-body proportion, "
    "oversized head, small rounded body, chubby rounded limbs, cute stylized proportions, "
    "{appearance}, the figure is painted in the clothing colours described above, "
    # 材质: 哑光 PVC / 软陶 / 搪胶
    "matte PVC vinyl material, soft clay texture, flocked vinyl toy surface, "
    "subtle fine surface texture, non-glossy finish, "
    # C4D 风格 3D 渲染 + 柔和顶侧光
    "C4D octane render, soft studio lighting, gentle top-side light, "
    "soft ambient occlusion, smooth gradient shading, "
    # 正面完整构图
    "front view, centered composition, full body visible, whole figure in frame, "
    "standing straight, arms relaxed at sides, both feet flat on the ground, "
    "static neutral standing pose, symmetrical pose, "
    "plain neutral studio backdrop, soft contact shadow, "
    # 产品摄影质感
    "product photography, sharp focus, high detail"
)

# 负向提示词，主要用于限制过度写实的人体特征、复杂背景和杂乱元素
NEGATIVE_PROMPT = (
    # 限制过度写实的皮肤和人体比例
    "photorealistic, realistic photo, real human, adult proportions, "
    "normal head-to-body ratio, tall slender body, "
    # 限制过长四肢和异常头身比例
    "long legs, long arms, elongated limbs, tiny head, "
    # 限制五官畸形和眼睛不对称
    "deformed face, distorted face, asymmetric eyes, cross-eyed, "
    "extra eyes, mutated hands, "
    # 限制多余手指、缺失手指和异常肢体
    "extra fingers, missing fingers, fused fingers, extra limbs, missing limbs, "
    "extra arms, extra legs, bad anatomy, "
    "extra hands, multiple hands, three hands, duplicated hands, duplicated limbs, "
    "malformed hands, mutated hands, misplaced hands, "
    # 限制过强塑料反光
    "glossy plastic, shiny surface, strong specular highlights, harsh reflections, "
    # 限制复杂背景和硬阴影
    "complex background, cluttered background, busy background, "
    "gradient background, vignette, dark background, textured background, "
    "patterned background, backdrop props, studio props, "
    "hard shadows, harsh lighting, dramatic lighting, "
    # 限制主体裁切
    "cropped, out of frame, cut off head, cut off feet, close-up, "
    # 限制模糊、噪点和低质量输出
    "blurry, out of focus, noise, grainy, jpeg artifacts, low quality, lowres, "
    # 限制文字、水印和品牌 Logo
    "watermark, text, signature, logo, brand logo, artist name, "
    "text on shirt, printed text on clothes, lettering, words, caption, label, "
    "gibberish text, garbled writing, "
    "badge, patch, emblem, sticker, tag, sew-on label, "
    "walking, running, jumping, raised leg, dynamic pose, tilted body, "
    # 真实人像常见干扰
    "skin pores, skin blemish, realistic skin texture, photograph of a person"
)


def build_positive(appearance: str) -> str:
    """把人物外观描述填入正向模板

    Args:
        appearance: 需要保留的发型、发色、发饰、服装款式与主色描述。
                    例如 "short black hair, round glasses, red hoodie"。
    """
    return POSITIVE_PROMPT.format(appearance=appearance.strip().rstrip(","))

DEFAULT_APPEARANCE = "short black hair, casual grey t-shirt"
