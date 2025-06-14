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
};

float3 getDiffuse(float3 normal, float3 directionTowardsLight) {
    float3 n = normal;
    float3 l = directionTowardsLight;
    
    float NdotL = max(dot(n, l), 0.0);

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

vertex VertexOut vs(VertexIn inVertex [[stage_in]],
                    uint vertexIndex [[vertex_id]],
                    constant float3 *positions [[buffer(1)]],
                    constant float3 *normals [[buffer(2)]],
                    constant float2 *uv [[buffer(3)]],
                    constant float4x4& translationModelMatrix [[buffer(4)]],
                    constant float4x4& rotationModelMatrix [[buffer(5)]],
                    constant float4x4& scaleModelMatrix [[buffer(6)]],
                    constant float4x4& finalModelMatrix [[buffer(7)]])
{
    VertexOut out;

    out.position = finalModelMatrix * float4(positions[vertexIndex], 1.0);
    float4x4 matrixWithoutProjection = translationModelMatrix * rotationModelMatrix * scaleModelMatrix;
    float3x3 normalMatrix = transpose(inverse(float3x3(
        matrixWithoutProjection.columns[0].xyz,
        matrixWithoutProjection.columns[1].xyz,
        matrixWithoutProjection.columns[2].xyz
    )));
    out.normal = normalize(normalMatrix * normals[vertexIndex]);
    out.uv = uv[vertexIndex];
    
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
    constant Material& u_material [[buffer(7)]])
{
    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float4 fillTextureOrAlbedoTexture = fillTexture.sample(s, outVertex.uv);
    
    float3 P = outVertex.position.xyz / outVertex.position.w;
    float3 L;
    float d;
    if (u_light.isOmni) {
        float3 lightVec = u_light.position - P;
        d = length(lightVec);
        L = normalize(lightVec);
    } else {
        L = -normalize(u_light.direction);
        d = 1.0;
    }

    float attenuation = 1.0 / (u_light.kc + u_light.kl * d + u_light.kq * d * d);

    float3 normal = normalize(outVertex.normal);
    float3 diffuse = getDiffuse(normal, L);
    float3 lightDiffuseColor = u_light.diffuseColor;
    float3 finalDiffuse = diffuse * lightDiffuseColor;

    float3 viewDirection = normalize(-outVertex.position.xyz);
    float3 finalSpecular = getSpecular(normal, L, viewDirection, u_material.shininess) * u_material.specularColor * u_light.specularIntensity;

    float3 finalColor = (finalDiffuse * u_material.diffuseColor + finalSpecular) * attenuation;

    return float4(finalColor * fillTextureOrAlbedoTexture.rgb, fillTextureOrAlbedoTexture.a);
}
