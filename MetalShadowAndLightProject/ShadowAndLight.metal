//
//  ShadowAndLight.metal
//  MetalShadowAndLightProject
//
//  Created by Michaël ATTAL on 16/04/2025.
//

#include <metal_stdlib>
using namespace metal;

// Not really necessary since we make a vertex buffer for each attribute instead, but let's keep it here just in case.
struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float2 uv;
    float4 shadowPosition;
    float3 worldPosition;
    float3 reflectDir;
};

struct Light {
    float3 position;
    float3 direction;
    float3 diffuseColor;
    float3 specularColor;
    float specularIntensity;
    float kc;
    float kl;
    float kq;
    bool isOmni;
};

struct Material {
    float3 diffuseColor;
    float3 specularColor;
    float shininess;
    bool isGround;
};

float3 getDiffuse(float3 normal, float3 directionTowardsLight) {
    float NdotL = max(dot(normal, directionTowardsLight), 0.0);
    return float3(NdotL);
}

float3 getReflect(float3 incident, float3 normal) {
    // Reflect the incident vector around the normal
    return normalize(reflect(-incident, normal)); // Metal’s reflect assumes I = direction from light
}

float3 getSpecular(float3 normal, float3 directionTowardsLight, float3 viewDirection, float shininess) {
    float3 H = normalize(directionTowardsLight + viewDirection); // Half-vector
    float NdotH = max(dot(normal, H), 0.0);
    return pow(NdotH, shininess);
}

// Inverse exist only for matrix 4x4 in metal
// Copied from https://developer.apple.com/forums/thread/722849?answerId=826064022#826064022
static float3x3 inverse(float3x3 const m)
{
    float const A =   m[1][1] * m[2][2] - m[2][1] * m[1][2];
    float const B = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]);
    float const C =   m[0][1] * m[1][2] - m[1][1] * m[0][2];
    float const D = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);
    float const E =   m[0][0] * m[2][2] - m[2][0] * m[0][2];
    float const F = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]);
    float const G =   m[1][0] * m[2][1] - m[2][0] * m[1][1];
    float const H = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]);
    float const I =   m[0][0] * m[1][1] - m[1][0] * m[0][1];
        
    float const det = m[0][0] * A + m[1][0] * B + m[2][0] * C;
    float const inv_det = 1.f / det;
    return inv_det * float3x3{
        float3{A, B, C},
        float3{D, E, F},
        float3{G, H, I}
    };
}

inline float PCF3x3(depth2d<float> depthTex,
                    sampler cmpSampler,
                    float3 sc)            // sc = UV.xy, depth.z
{
    float2 texel = 1.0 / float2(depthTex.get_width(),
                                depthTex.get_height());
    float acc = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 offset = float2(x, y) * texel;
            acc += depthTex.sample_compare(cmpSampler,
                                           sc.xy + offset,
                                           sc.z);
        }
    }
    return acc / 9.0;
}

vertex VertexOut vs(VertexIn inVertex [[stage_in]],
                    uint vertexIndex [[vertex_id]],
                    constant float3 *positions [[buffer(1)]],
                    constant float3 *normals [[buffer(2)]],
                    constant float2 *uv [[buffer(3)]],
                    constant float4x4& translationModelMatrix [[buffer(4)]],
                    constant float4x4& rotationModelMatrix [[buffer(5)]],
                    constant float4x4& scaleModelMatrix [[buffer(6)]],
                    constant float4x4& finalModelMatrix [[buffer(7)]],
                    constant float4x4& lightViewProjectionMatrix [[buffer(8)]])
{
    VertexOut out;

    out.position = finalModelMatrix * float4(positions[vertexIndex], 1.0);
    float4x4 matrixWithoutProjection = translationModelMatrix * rotationModelMatrix * scaleModelMatrix; // modelMatrix
    float3x3 normalMatrix = transpose(inverse(float3x3(
        matrixWithoutProjection.columns[0].xyz,
        matrixWithoutProjection.columns[1].xyz,
        matrixWithoutProjection.columns[2].xyz
    )));
    out.normal = normalize(normalMatrix * normals[vertexIndex]);
    out.uv = uv[vertexIndex];
    
    float4 worldPos = matrixWithoutProjection * float4(positions[vertexIndex], 1.0);
    out.worldPosition = worldPos.xyz;
    out.shadowPosition = lightViewProjectionMatrix * worldPos;
    out.reflectDir = reflect(normalize(worldPos.xyz), out.normal);
    return out;
}


fragment float4 fs(
    VertexOut outVertex [[stage_in]],
    texture2d<float> fillTexture [[texture(0)]],
    texture2d<float> displacementTexture [[texture(1)]],
    texture2d<float> normalTexture [[texture(2)]],
    texture2d<float> roughnessTexture [[texture(3)]],
    texture2d<float> HDRPmaskMapTexture [[texture(4)]],
    constant bool& useAllTextures [[buffer(5)]],
    constant Light& u_light [[buffer(6)]],
    constant Material& u_material [[buffer(7)]],
    depth2d<float> shadowTexture [[texture(8)]],
    texturecube<float> skybox [[texture(9)]])
{
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    constexpr sampler shadowSampler(coord::normalized,
                                    filter::linear,
                                    address::clamp_to_border,
                                    compare_func::less_equal,
                                    border_color::opaque_white);
    
    float4 fillTextureOrAlbedoTexture = fillTexture.sample(s, outVertex.uv);
    
    float3 P = outVertex.position.xyz / outVertex.position.w;

    float3 L = normalize(u_light.position - outVertex.worldPosition);
    if (!u_light.isOmni) L = -normalize(u_light.direction);
    
    float attenuation = 1.0;
    if(u_light.isOmni) {
        float d = length(u_light.position - outVertex.worldPosition);
        attenuation = 1.0 / (u_light.kc + u_light.kl*d + u_light.kq*d*d);
    }

    float3 normal = normalize(outVertex.normal);
    float3 diffuse = getDiffuse(normal, L);
    float3 lightDiffuseColor = u_light.diffuseColor;
    float3 finalDiffuse = diffuse * lightDiffuseColor;

    float3 viewDirection = normalize(-outVertex.worldPosition);
    float3 finalSpecular = getSpecular(normal, L, viewDirection, u_material.shininess) * u_material.specularColor * u_light.specularIntensity;

    float3 shadowCoord = outVertex.shadowPosition.xyz / outVertex.shadowPosition.w;
    shadowCoord        = shadowCoord * 0.5 + 0.5;        // NDC -> [0,1]
    // float bias = max(0.001 * (1.0 - dot(normal, L)), 0.0005);
    float texelSize = 1.0 / float(shadowTexture.get_width());
    float bias = max(4.0 * texelSize * (1.0 - dot(normal, L)),
                     1.5 * texelSize);
    // float visibility  = shadowTexture.sample_compare(shadowSampler, shadowCoord.xy, shadowCoord.z - bias);
    float visibility = PCF3x3(shadowTexture, shadowSampler, float3(shadowCoord.xy, shadowCoord.z - bias));

    // Small ambient term to avoid pitch-black
    visibility = 0.2 + 0.8 * visibility;
    
    float3 color = (finalDiffuse * u_material.diffuseColor + finalSpecular) * attenuation * visibility;
    if (u_material.isGround == false ) {
        color = (finalDiffuse * u_material.diffuseColor + finalSpecular) * attenuation; // Comment to see the shadow on other models in the scene (actually, it bug: the shadow is also shown on the model itself)
    }
    float3 reflectedColor = skybox.sample(s, outVertex.reflectDir).rgb;
    float3 finalColor = mix(color, reflectedColor, 0.3); // Partial specular reflection via Environment Mapping
    
    
    return float4(finalColor * fillTextureOrAlbedoTexture.rgb, fillTextureOrAlbedoTexture.a);
}

// Shadow vertex shader for depth rendering
vertex float4 shadow_vs(VertexIn inVertex [[stage_in]],
                        uint vertexIndex [[vertex_id]],
                        constant float3 *positions [[buffer(1)]],
                        constant float4x4& translationModelMatrix [[buffer(2)]],
                        constant float4x4& rotationModelMatrix [[buffer(3)]],
                        constant float4x4& scaleModelMatrix [[buffer(4)]],
                        constant float4x4& lightViewProjectionMatrix [[buffer(5)]])
{
    // return float4(); // Uncomment to show without shadow
    float4x4 modelMatrix = translationModelMatrix * rotationModelMatrix * scaleModelMatrix;
    float4 worldPos = modelMatrix * float4(positions[vertexIndex], 1.0);
    return lightViewProjectionMatrix * worldPos;
}

struct SkyboxOut {
    float4 position [[position]];
    float3 direction;
};

vertex SkyboxOut skybox_vs(uint vertexID [[vertex_id]])
{
    float3 cubeVertices[36] = {
        float3(-1, -1, -1), float3(1, -1, -1), float3(1,  1, -1),
        float3(1,  1, -1), float3(-1,  1, -1), float3(-1, -1, -1),
        float3(-1, -1,  1), float3(1, -1,  1), float3(1,  1,  1),
        float3(1,  1,  1), float3(-1,  1,  1), float3(-1, -1,  1),
        float3(-1,  1,  1), float3(-1,  1, -1), float3(-1, -1, -1),
        float3(-1, -1, -1), float3(-1, -1,  1), float3(-1,  1,  1),
        float3(1,  1,  1), float3(1,  1, -1), float3(1, -1, -1),
        float3(1, -1, -1), float3(1, -1,  1), float3(1,  1,  1),
        float3(-1, -1, -1), float3(1, -1, -1), float3(1, -1,  1),
        float3(1, -1,  1), float3(-1, -1,  1), float3(-1, -1, -1),
        float3(-1,  1, -1), float3(1,  1, -1), float3(1,  1,  1),
        float3(1,  1,  1), float3(-1,  1,  1), float3(-1,  1, -1)
    };

    SkyboxOut out;
    out.direction = cubeVertices[vertexID];
    out.position = float4(cubeVertices[vertexID], 1.0);
    return out;
}

fragment float4 skybox_fs(SkyboxOut in [[stage_in]],
                          texturecube<float> skybox [[texture(0)]])
{
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    return float4(skybox.sample(s, normalize(in.direction)).rgb, 1.0);
}
