//
//  ShadowAndLightMetalRenderer.swift
//  MetalShadowAndLightProject
//
//  Created by Michaël ATTAL on 16/04/2025.
//

import MetalKit
import ModelIO

struct Light {
    var position: SIMD3<Float>
    var direction: SIMD3<Float>
    var diffuseColor: SIMD3<Float>
    var specularColor: SIMD3<Float>
    var specularIntensity: Float
    var kc: Float
    var kl: Float
    var kq: Float
    var isOmni: Bool
}

struct Material {
    var diffuseColor: SIMD3<Float>
    var specularColor: SIMD3<Float>
    var shininess: Float
    var isGround: Bool = false
}

struct VertexIn {
    let position: SIMD3<Float>
    let normal: SIMD3<Float>
    let uv: SIMD2<Float>
}

enum Direction {
    case top_right
    case bottom_left
}

enum ScaleEffect {
    case grow
    case shrink
}

final class ShadowAndLightMetalRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private var pipelineState: MTLRenderPipelineState?
    private var commandQueue: MTLCommandQueue?
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    private var positionsBuffer: MTLBuffer?
    private var normalsBuffer: MTLBuffer?
    private var direction: Direction = .top_right
    private var scaleEffect: ScaleEffect = .grow
    
    private var projectionMatrix: matrix_float4x4 = .init()
    private var translationMatrix: matrix_float4x4 = .init()
    private var rotationMatrix: matrix_float4x4 = .init()
    private var scaleMatrix: matrix_float4x4 = .init()
    private var finalMatrix: matrix_float4x4 = .init()
    
    private var fillTexture: MTLTexture?
    private var displacementTexture: MTLTexture?
    private var normalTexture: MTLTexture?
    private var roughnessTexture: MTLTexture?
    private var HDRPmaskMapTexture: MTLTexture?
    private var textureData: [UInt8] = []
    private var uvBuffer: MTLBuffer?
    private var uvTexture: [SIMD2<Float>] = []
    private var textureWidth: Int = 1
    private var textureHeight: Int = 1

    private var depthStencilState: MTLDepthStencilState?
    
    private var indices: [UInt16] = []
    
    private let nbIndicesDragon = DragonIndices.count
    private let nbVerticesDragon = DragonVertices.count / 8 // because each vertex contains X,Y,Z, NX, NY, NZ, U, V = 8 floats per vertex
    
    private var isObj = false
    private var objMtlVertexDescriptor: MTLVertexDescriptor?
    private var objURL: URL?
    private var objMesh: MTKMesh?
    private var objSubmesh: MTKSubmesh?

    private var light = Light(
        position: SIMD3<Float>(0, 10, 30),
        direction: SIMD3<Float>(0, -1, -1),
        diffuseColor: SIMD3<Float>(1, 1, 1),
        specularColor: SIMD3<Float>(1, 1, 1),
        specularIntensity: 1.0,
        kc: 1.0, kl: 0.09, kq: 0.032,
        isOmni: false
    )
    private var material = Material(diffuseColor: SIMD3<Float>(1, 1, 1), specularColor: SIMD3<Float>(1, 1, 1), shininess: 64)
    
    private var shadowPipelineState: MTLRenderPipelineState?
    private var skyboxPipelineState: MTLRenderPipelineState?
    
    private var shadowTexture: MTLTexture?
    private var skyboxTexture: MTLTexture?
    
    private var viewMatrix = matrix_float4x4()
    private var modelMatrix = matrix_float4x4()
    private var lightViewProjectionMatrix = matrix_float4x4()
    
    // Plane mesh buffers and indices
    private var planeVertexBuffer: MTLBuffer?
    private var planeIndexBuffer: MTLBuffer?
    private var planePositionsBuffer: MTLBuffer?
    private var planeNormalsBuffer: MTLBuffer?
    private var planeUVBuffer: MTLBuffer?
    private var planeIndices: [UInt16] = []
    
    private var planeMaterial = Material(diffuseColor: SIMD3<Float>(1, 1, 1), specularColor: SIMD3<Float>(0.1, 0.1, 0.1), shininess: 8, isGround: true)
    private var planeFillTexture: MTLTexture?

    init(mtkView: MTKView, objURL: URL? = nil) {
        guard let device = mtkView.device else {
            fatalError("MTKView has no MTLDevice.")
        }
        self.device = device
        super.init()
        
        self.objURL = objURL
        
        // Creates CommandQueue and pipeline
        buildResources(mtkView: mtkView)
    }
    
    // MARK: - Resource Setup

    private func buildResources(mtkView: MTKView) {
        setupMeshData()
        setupPlaneMesh()
        setupPipeline(mtkView: mtkView)
        setupDepthState()
        setupTexture()
        setupFillTexture()
        setupShadowTexture()
        setupSkyboxTexture()
        setupWhiteTextureForPlane()
    }
    
    // MARK: - Mesh Setup
    
    private func setupMeshData() {
        if let objURL = objURL {
            isObj = true
            setupOBJMesh(useRandomUVs: false) // No texture for the bear model found in internet, so I use random UVs
        } else {
            isObj = false
            setupDragonMesh()
        }
    }
    
    private func setupOBJMesh(useRandomUVs: Bool = false) {
        // 1. Prepare ModelIO vertex descriptor for position, normal, uv
        let mdlVertexDescriptor = MDLVertexDescriptor()
        mdlVertexDescriptor.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition,
                                                               format: .float3,
                                                               offset: 0,
                                                               bufferIndex: 0)
        mdlVertexDescriptor.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal,
                                                               format: .float3,
                                                               offset: MemoryLayout<SIMD3<Float>>.stride,
                                                               bufferIndex: 0)
        mdlVertexDescriptor.attributes[2] = MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate,
                                                               format: .float2,
                                                               offset: MemoryLayout<SIMD3<Float>>.stride * 2,
                                                               bufferIndex: 0)
        mdlVertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: MemoryLayout<VertexIn>.stride)
        
        // 2. Load the OBJ using ModelIO with custom descriptor
        let allocator = MTKMeshBufferAllocator(device: device)
        let asset = MDLAsset(url: objURL,
                             vertexDescriptor: mdlVertexDescriptor,
                             bufferAllocator: allocator)
        guard let mdlMesh = asset.childObjects(of: MDLMesh.self).first as? MDLMesh else {
            fatalError("Failed to load MDLMesh from URL: \(objURL)")
        }
        mdlMesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal,
                           creaseThreshold: 0.0)
        
        // Read Blinn-Phong material parameters from MTL if present
        if let mdlMaterial = mdlMesh.submeshes?.firstObject as? MDLSubmesh, let material = mdlMaterial.material {
            // Diffuse color
            if let diffuse = material.property(with: .baseColor) {
                if diffuse.type == .float3 {
                    self.material.diffuseColor = SIMD3<Float>(diffuse.float3Value)
                } else if diffuse.type == .float4 {
                    let color = diffuse.float4Value
                    self.material.diffuseColor = SIMD3<Float>(color.x, color.y, color.z)
                }
            }
            // Specular color
            if let specular = material.property(with: .specular) {
                if specular.type == .float3 {
                    self.material.specularColor = SIMD3<Float>(specular.float3Value)
                } else if specular.type == .float4 {
                    let color = specular.float4Value
                    self.material.specularColor = SIMD3<Float>(color.x, color.y, color.z)
                }
            }
            // Shininess
            if let shininess = material.property(with: .specularExponent) {
                if shininess.type == .float {
                    let s = shininess.floatValue
                    self.material.shininess = max(1, s * 0.25)
                }
            }
        }
        
        // 3. Create an MTKMesh from the ModelIO mesh
        do {
            objMesh = try MTKMesh(mesh: mdlMesh, device: device)
        } catch {
            fatalError("Failed to create MTKMesh: \(error)")
        }
        guard let objMesh = objMesh, let firstSubmesh = objMesh.submeshes.first else {
            fatalError("No submeshes found in the OBJ model.")
        }
        objSubmesh = firstSubmesh
        
        // Convert MDL to MTL vertex descriptor for pipeline
        guard let mtlVertexDescriptor = MTKMetalVertexDescriptorFromModelIO(mdlVertexDescriptor) else {
            fatalError("Failed to convert MDLVertexDescriptor to MTLVertexDescriptor.")
        }
        
        objMtlVertexDescriptor = mtlVertexDescriptor
        
        guard let desc = objMtlVertexDescriptor else {
            fatalError("MTLVertexDescriptor from obj setup is nil")
        }
        
        guard let vb = objMesh.vertexBuffers.first else {
            fatalError("No vertex buffers found in the OBJ model.")
        }
        
        let stride = desc.layouts[0].stride
        let posOffset = desc.attributes[0].offset
        let normalOffset = desc.attributes[1].offset
        let uvOffset = desc.attributes[2].offset
        let vertexCount = objMesh.vertexCount
        let rawPtr = vb.buffer.contents()

        var positions = [SIMD3<Float>]()
        var normals = [SIMD3<Float>]()
        var uvs = [SIMD2<Float>]()
        positions.reserveCapacity(vertexCount)
        normals.reserveCapacity(vertexCount)
        uvs.reserveCapacity(vertexCount)

        for i in 0 ..< vertexCount {
            let basePtr = rawPtr.advanced(by: i * stride)
            // Positions
            let pPtr = basePtr.advanced(by: posOffset).assumingMemoryBound(to: Float.self)
            let p = SIMD3<Float>(pPtr[0], pPtr[1], pPtr[2])
            positions.append(p)
            // Normals
            let nPtr = basePtr.advanced(by: normalOffset).assumingMemoryBound(to: Float.self)
            let n = SIMD3<Float>(nPtr[0], nPtr[1], nPtr[2])
            normals.append(n)
            // UVs
            let uPtr = basePtr.advanced(by: uvOffset).assumingMemoryBound(to: Float.self)
            let u = SIMD2<Float>(uPtr[0], uPtr[1])
            uvs.append(u)
        }

        if useRandomUVs {
            // Get random uv for better visual effect
            for i in 0 ..< vertexCount {
                if uvs[i].x == 0 && uvs[i].y == 0 {
                    uvs[i].x = Float.random(in: 0...1)
                    uvs[i].y = Float.random(in: 0...1)
                }
            }
        }
        
        positionsBuffer = device.makeBuffer(
            bytes: positions,
            length: positions.count * MemoryLayout<SIMD3<Float>>.stride,
            options: []
        )
        normalsBuffer = device.makeBuffer(
            bytes: normals,
            length: normals.count * MemoryLayout<SIMD3<Float>>.stride,
            options: []
        )
        uvBuffer = device.makeBuffer(
            bytes: uvs,
            length: uvs.count * MemoryLayout<SIMD2<Float>>.stride,
            options: []
        )

        if let sub = objSubmesh {
            indexBuffer = sub.indexBuffer.buffer
            indices = Array(UnsafeBufferPointer(start: sub.indexBuffer.buffer.contents().assumingMemoryBound(to: UInt16.self),
                                                count: Int(sub.indexCount)))
        }
                    
        vertexBuffer = objMesh.vertexBuffers[0].buffer
        isObj = true
    }
    
    private func setupDragonMesh() {
        var positions3D: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var uvs: [SIMD2<Float>] = []
        
        for i in stride(from: 0, to: DragonVertices.count, by: 8) {
            positions3D.append([DragonVertices[i], DragonVertices[i + 1], DragonVertices[i + 2]])
            normals.append([DragonVertices[i + 3], DragonVertices[i + 4], DragonVertices[i + 5]])
            uvs.append([DragonVertices[i + 6], DragonVertices[i + 7]])
        }
        
        var vertices: [VertexIn] = []
        vertices.reserveCapacity(positions3D.count)
        for (i, p) in positions3D.enumerated() {
            vertices.append(VertexIn(position: p, normal: normals[i], uv: uvs[i]))
        }
        
        vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<VertexIn>.stride,
            options: []
        )
        
        // Define an array of indices for the dragon
        indices = DragonIndices
        
        // Create the index buffer
        indexBuffer = device.makeBuffer(
            bytes: indices,
            length: indices.count * MemoryLayout<UInt16>.stride,
            options: []
        )
        
        positionsBuffer = device.makeBuffer(
            bytes: positions3D,
            length: positions3D.count * MemoryLayout<SIMD3<Float>>.stride,
            options: []
        )
        
        normalsBuffer = device.makeBuffer(
            bytes: normals,
            length: normals.count * MemoryLayout<SIMD3<Float>>.stride,
            options: []
        )
        
        uvTexture = uvs
        uvBuffer = device.makeBuffer(
            bytes: uvTexture,
            length: uvTexture.count * MemoryLayout<SIMD2<Float>>.stride,
            options: []
        )
    }
    
    private func setupPlaneMesh(width: Float = 50, depth: Float = 50, y: Float = -3.0) {
        // Four vertices for a simple plane (two triangles)
        let positions: [SIMD3<Float>] = [
            SIMD3<Float>(-width / 2, y, -depth / 2),
            SIMD3<Float>(width / 2, y, -depth / 2),
            SIMD3<Float>(width / 2, y, depth / 2),
            SIMD3<Float>(-width / 2, y, depth / 2),
        ]
        let normals: [SIMD3<Float>] = [
            SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(0, 1, 0),
            SIMD3<Float>(0, 1, 0),
        ]
        let uvs: [SIMD2<Float>] = [
            SIMD2<Float>(0, 0),
            SIMD2<Float>(1, 0),
            SIMD2<Float>(1, 1),
            SIMD2<Float>(0, 1),
        ]
        let indices: [UInt16] = [
            0, 1, 2,
            2, 3, 0,
        ]

        planePositionsBuffer = device.makeBuffer(bytes: positions, length: positions.count * MemoryLayout<SIMD3<Float>>.stride, options: [])
        planeNormalsBuffer = device.makeBuffer(bytes: normals, length: normals.count * MemoryLayout<SIMD3<Float>>.stride, options: [])
        planeUVBuffer = device.makeBuffer(bytes: uvs, length: uvs.count * MemoryLayout<SIMD2<Float>>.stride, options: [])
        planeIndices = indices
        planeIndexBuffer = device.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt16>.stride, options: [])

        var verts: [VertexIn] = []
        for i in 0 ..< positions.count {
            verts.append(VertexIn(position: positions[i], normal: normals[i], uv: uvs[i]))
        }
        planeVertexBuffer = device.makeBuffer(bytes: verts, length: verts.count * MemoryLayout<VertexIn>.stride, options: [])
    }
    
    // MARK: - Pipeline Setup

    private func setupPipeline(mtkView: MTKView) {
        mtkView.depthStencilPixelFormat = .depth32Float
        
        commandQueue = device.makeCommandQueue()
        
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to create Metal library.")
        }
        
        // Main pipeline
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "vs")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "fs")
        pipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        
        // Not used for the moment, we pass all these 3 attributes via vertex buffers
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float3
        vertexDescriptor.attributes[1].offset = 16 // MemoryLayout<SIMD3<Float>>.stride // All SIMD count as 16 bytes for optimization (it adds padding)
        vertexDescriptor.attributes[2].bufferIndex = 0
        vertexDescriptor.attributes[2].format = .float2
        vertexDescriptor.attributes[2].offset = 16
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<VertexIn>.stride
        
        pipelineDescriptor.vertexDescriptor = isObj ? objMtlVertexDescriptor : vertexDescriptor
        
        // Configure depth format for the pipeline to match the view's format
        pipelineDescriptor.depthAttachmentPixelFormat = mtkView.depthStencilPixelFormat
        
        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Could not create pipeline state: \(error)")
        }
        
        // Shadow pipeline
        let shadowDesc = MTLRenderPipelineDescriptor()
        shadowDesc.vertexDescriptor = isObj ? objMtlVertexDescriptor : vertexDescriptor
        shadowDesc.vertexFunction = library.makeFunction(name: "shadow_vs")
        shadowDesc.fragmentFunction = nil
        shadowDesc.depthAttachmentPixelFormat = .depth32Float
        do {
            shadowPipelineState = try device.makeRenderPipelineState(descriptor: shadowDesc)
        } catch {
            fatalError("Could not create shadow pipeline state: \(error)")
        }

        // Skybox pipeline
        let skyboxDesc = MTLRenderPipelineDescriptor()
        skyboxDesc.vertexDescriptor = nil
        skyboxDesc.vertexFunction = library.makeFunction(name: "skybox_vs")
        skyboxDesc.fragmentFunction = library.makeFunction(name: "skybox_fs")
        skyboxDesc.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        skyboxDesc.depthAttachmentPixelFormat = .depth32Float
        do {
            skyboxPipelineState = try device.makeRenderPipelineState(descriptor: skyboxDesc)
        } catch {
            fatalError("Could not create skybox pipeline state: \(error)")
        }
    }
    
    // MARK: - Depth State Setup

    private func setupDepthState() {
        let depthDescriptor = MTLDepthStencilDescriptor() // Configure Z-buffer
        depthDescriptor.isDepthWriteEnabled = true
        depthDescriptor.depthCompareFunction = .less
        depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)
    }
    
    private func setupShadowTexture() {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 1024, height: 1024, mipmapped: false)
        desc.usage = [.shaderRead, .renderTarget]
        shadowTexture = device.makeTexture(descriptor: desc)
    }

    private func setupSkyboxTexture() {
        let names = ["pisa_posx", "pisa_negx", "pisa_posy", "pisa_negy", "pisa_posz", "pisa_negz"]

        guard let mdlTexture = MDLTexture(cubeWithImagesNamed: names, bundle: Bundle.main) else {
            print("Error: Impossible to load the MDLTEXTURE for the SkyBox.")
            skyboxTexture = nil
            return
        }

        let textureLoader = MTKTextureLoader(device: device)

        do {
            skyboxTexture = try textureLoader.newTexture(texture: mdlTexture, options: [
                .origin: MTKTextureLoader.Origin.topLeft,
                .SRGB: false,
            ])
        } catch {
            print("Error when loading the cubemap: \(error)")
            skyboxTexture = nil
        }
    }

    private func setupFillTexture() {
        fillTexture = loadPrebuiltTexture(name: isObj ? "monkey_albedo" : "dragon")
    }

    // MARK: - Texture Setup Functions

    private func setupTexture() {
        // Configure texture dimensions
        textureWidth = 4
        textureHeight = 4
        
        // Create texture descriptor
        let desc = MTLTextureDescriptor()
        desc.pixelFormat = .rgba8Unorm
        desc.width = textureWidth
        desc.height = textureHeight
        desc.usage = [.shaderRead, .renderTarget]
        desc.storageMode = .managed
        fillTexture = device.makeTexture(descriptor: desc)
        
        // Different texture options (uncomment one to use):
        fillTexture = loadPrebuiltTexture(name: isObj ? "monkey_albedo" : "dragon") // TODO: pass it as parameter also later
        if isObj {
            displacementTexture = loadPrebuiltTexture(name: "monkey_displacement")
            normalTexture = loadPrebuiltTexture(name: "monkey_normal")
            roughnessTexture = loadPrebuiltTexture(name: "monkey_roughness")
            HDRPmaskMapTexture = loadPrebuiltTexture(name: "monkey_HDRP_mask_map")
        }
        
        // Option 1: Uniform color (green dragon)
        // textureData = updateTextureData(useUniformColor: true, rgbaColor: [0, 255, 0, 255])
        // fillTexture = updateFillTexture(textureData)
        
        // Option 2: Multiple colors
        // textureData = updateTextureData(useUniformColor: false)
        // fillTexture = updateFillTexture(textureData)
        
        // Option 3: Custom texture (not working correctly currently, I use the prebuilt metal texture loader for now)
        // textureData = updateTextureData(useCustomTexture: true, customTextureName: "dragon")
        // fillTexture = updateFillTexture(textureData)
    }
    
    private func loadPrebuiltTexture(name: String) -> MTLTexture? {
        var newTexture: MTLTexture?
        let textureLoader = MTKTextureLoader(device: device)
        if let url = Bundle.main.url(forResource: name, withExtension: "png") {
            do {
                newTexture = try textureLoader.newTexture(URL: url, options: [
                    MTKTextureLoader.Option.origin: MTKTextureLoader.Origin.bottomLeft,
                    MTKTextureLoader.Option.SRGB: false,
                    MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.managed.rawValue),
                ])
            } catch {
                print("Failed to load custom texture: \(error)")
            }
        } else {
            print("Texture \(name).png not found in bundle.")
        }
        return newTexture
    }
    
    private func updateTextureData(
        useUniformColor: Bool = false,
        rgbaColor: SIMD4<UInt8> = [0, 255, 0, 255],
        useCustomTexture: Bool = false,
        customTextureName: String = "my-texture"
    ) -> [UInt8] {
        if useCustomTexture {
            return loadCustomTextureData(name: customTextureName)
        } else if useUniformColor || textureWidth == 1 || textureHeight == 1 {
            return createUniformColorData(color: rgbaColor)
        } else {
            return createRandomColorData()
        }
    }
    
    private func loadCustomTextureData(name: String) -> [UInt8] {
        guard
            let url = Bundle.main.url(forResource: name, withExtension: "png"),
            let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
        else {
            print("Failed to load or decode PNG image: \(name).png")
            return []
        }
        
        let width = min(cgImage.width, textureWidth)
        let height = min(cgImage.height, textureHeight)
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bitsPerComponent = 8
        
        var textureData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        guard
            let context = CGContext(
                data: &textureData,
                width: width,
                height: height,
                bitsPerComponent: bitsPerComponent,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else {
            print("Failed to create CGContext for texture image.")
            return []
        }
        
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        context.clear(rect)
        context.draw(cgImage, in: rect)
        
        return textureData
    }
    
    private func createUniformColorData(color: SIMD4<UInt8>) -> [UInt8] {
        var data = [UInt8](repeating: 0, count: textureWidth * textureHeight * 4)
        for i in 0 ..< (textureWidth * textureHeight) {
            let idx = i * 4
            data[idx + 0] = color.x
            data[idx + 1] = color.y
            data[idx + 2] = color.z
            data[idx + 3] = color.w
        }
        return data
    }
    
    private func createRandomColorData() -> [UInt8] {
        var data = [UInt8](repeating: 0, count: textureWidth * textureHeight * 4)
        for i in 0 ..< (textureWidth * textureHeight) {
            let idx = i * 4
            data[idx + 0] = UInt8.random(in: 0 ..< 255)
            data[idx + 1] = UInt8.random(in: 0 ..< 255)
            data[idx + 2] = UInt8.random(in: 0 ..< 255)
            data[idx + 3] = 255
        }
        return data
    }
    
    private func updateFillTexture(_ textureBuffer: [UInt8]) -> MTLTexture? {
        guard let texture = fillTexture else { return nil }
        texture.replace(
            region: MTLRegionMake2D(0, 0, textureWidth, textureHeight),
            mipmapLevel: 0,
            withBytes: textureBuffer,
            bytesPerRow: textureWidth * 4
        )
        return texture
    }
    
    private func setupWhiteTextureForPlane() {
        let white: [UInt8] = [255, 255, 255, 255]
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else { return }
        tex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: white, bytesPerRow: 4)
        planeFillTexture = tex
    }
    
    // MARK: - Matrix Setup

    private func setupMatrices() {
        projectionMatrix = getProjectionMatrix()
        translationMatrix = getNewTranslationMatrix(isInitialState: true)
        rotationMatrix = getNewRotationMatrix(isInitialState: true, for: 0.0)
        scaleMatrix = getNewScaleMatrix(isInitialState: true)
        finalMatrix = projectionMatrix * translationMatrix * rotationMatrix * scaleMatrix
    }
    
    // MARK: - Matrix Creation Functions

    func getProjectionMatrix(near: Float = 0.1, far: Float = 1000.0, aspect: Float = 1.0) -> float4x4 {
        let fovyRadians: Float = .pi / 4.0 // 45 degrees
        let cotangent = 1.0 / tan(fovyRadians / 2)
        let projectionMatrix = matrix_float4x4.init(
            rows: [
                // C0  C1   C2   C3
                [cotangent / aspect, 0.0, 0.0, 0.0],
                [0.0, cotangent, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), 2 * near * far / (near - far)],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )
        return projectionMatrix
    }
    
    func getNewTranslationMatrix(isInitialState: Bool = false) -> float4x4 {
        if isInitialState {
            let translationMatrix = matrix_float4x4.init(
                rows: [
                    // C0  C1   C2   C3
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, isObj ? 0.0 : -3.0], // Down the dragon a little bit
                    [0.0, 0.0, 1.0, isObj ? -5.0 : -30.0], // Since the implementation of the projection matrix, move the dragon a little bit.
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            return translationMatrix
        } else {
            let prevTx = translationMatrix.columns.3.x
            let prevTy = translationMatrix.columns.3.y
            if prevTx >= 5.0 {
                direction = .bottom_left
            } else if prevTy <= -5.0 {
                direction = .top_right
            }
            // For fun: Switch back to initial state to do a ping pong effect :D
            let tx: Float = direction == .top_right ? prevTx + 0.01 : prevTx - 0.01
            let ty: Float = direction == .top_right ? prevTy + 0.01 : prevTy - 0.01
            let tz: Float = isObj ? -5.0 : -30.0
            let translationMatrix = matrix_float4x4.init(
                rows: [
                    // C0  C1   C2   C3
                    [1.0, 0.0, 0.0, tx],
                    [0.0, 1.0, 0.0, ty],
                    [0.0, 0.0, 1.0, tz],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            return translationMatrix
        }
    }
    
    func getNewRotationMatrix(isInitialState: Bool = false, for time: TimeInterval) -> float4x4 {
        if isInitialState {
            let rotationMatrix = matrix_float4x4.init(
                rows: [
                    // C0  C1   C2   C3
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            return rotationMatrix
        } else {
            let angle = Float(time)
            let cA = cos(angle) // cosAngle
            let sA = sin(angle) // sinAngle

            // Rotate on X
            let rotationMatrix = matrix_float4x4.init(
                // x
                // rows: [
                //     // C0  C1   C2   C3
                //     [1.0, 0.0, 0.0, 0.0],
                //     [0.0, cA, -sA, 0.0],
                //     [0.0, sA, cA, 0.0],
                //     [0.0, 0.0, 0.0, 1.0],
                // ]
                // y
                rows: [
                    // C0  C1   C2   C3
                    [cA, 0.0, -sA, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [sA, 0.0, cA, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
                // z
                // rows: [
                //     // C0  C1   C2   C3
                //     [cA, -sA, 0.0, 0.0],
                //     [sA, cA, 0.0, 0.0],
                //     [0.0, 0.0, 1.0, 0.0],
                //     [0.0, 0.0, 0.0, 1.0],
                // ]
            )
            
            return rotationMatrix
        }
    }

    func getNewScaleMatrix(isInitialState: Bool = false) -> float4x4 {
        if isInitialState {
            let scaleMatrix = matrix_float4x4.init(
                rows: [
                    // C0  C1   C2   C3
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            return scaleMatrix
        } else {
            let prevSx = scaleMatrix.columns.0.x
            let prevSy = scaleMatrix.columns.1.y
            let prevSz = scaleMatrix.columns.2.z
            if prevSx >= 2.0 {
                scaleEffect = .shrink
            } else if prevSx <= 0.5 {
                scaleEffect = .grow
            }
            // For fun: grow/shrink effect
            let sx: Float = scaleEffect == .grow ? prevSx + 0.01 : prevSx - 0.01
            let sy: Float = scaleEffect == .grow ? prevSy + 0.01 : prevSy - 0.01
            let sz: Float = scaleEffect == .grow ? prevSz + 0.01 : prevSz - 0.01
            let scaleMatrix = matrix_float4x4.init(
                rows: [
                    // C0  C1   C2   C3
                    [sx, 0.0, 0.0, 0.0],
                    [0.0, sy, 0.0, 0.0],
                    [0.0, 0.0, sz, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            return scaleMatrix
        }
    }
    
    // MARK: - Matrix utils for Shadow Mapping

    func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
        let f = normalize(center - eye)
        let s = normalize(cross(f, up))
        let u = cross(s, f)
        
        let m = float4x4([
            SIMD4<Float>(s.x, u.x, -f.x, 0.0),
            SIMD4<Float>(s.y, u.y, -f.y, 0.0),
            SIMD4<Float>(s.z, u.z, -f.z, 0.0),
            SIMD4<Float>(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0),
        ])
        return m.transpose
    }

    func orthographicProjection(left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) -> float4x4 {
        let rl = right - left
        let tb = top - bottom
        let fn = far - near
        
        let m = float4x4([
            SIMD4<Float>(2.0 / rl, 0, 0, 0),
            SIMD4<Float>(0, 2.0 / tb, 0, 0),
            SIMD4<Float>(0, 0, -1.0 / fn, 0),
            SIMD4<Float>(-(right + left) / rl, -(top + bottom) / tb, -near / fn, 1),
        ])
        return m
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // TODO: Handle view resize here
    }
    
    /// Called every frame to draw
    func draw(in view: MTKView) {
        guard
            let pipelineState = pipelineState,
            let commandQueue = commandQueue,
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let passDescriptor = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable,
            let vertexBuffer = vertexBuffer,
            let indexBuffer = indexBuffer,
            let positionsBuffer = positionsBuffer,
            let uvBuffer = uvBuffer,
            let depthStencilState = depthStencilState,
            let normalsBuffer = normalsBuffer,
            indices.count > 0
        else {
            return
        }
        // Update matrices
        projectionMatrix = getProjectionMatrix()
        translationMatrix = getNewTranslationMatrix(isInitialState: true) // Set isInitialState to false to see ping pong effect
        rotationMatrix = getNewRotationMatrix(isInitialState: false, for: CACurrentMediaTime()) // Set isInitialState to false to see rotation effect

        scaleMatrix = getNewScaleMatrix(isInitialState: true) // Set isInitialState to false to see grow/shrink effect
        finalMatrix = projectionMatrix * translationMatrix * rotationMatrix * scaleMatrix // TRS
        
        let sceneRadius: Float = 15
        let sceneHeight: Float = 5

        let left = -sceneRadius
        let right = sceneRadius
        let bottom = -sceneRadius
        let top = sceneRadius
        let near: Float = 10
        let far: Float = sceneRadius + sceneHeight + length(light.position)

        let lightProj = orthographicProjection(left: left, right: right,
                                               bottom: bottom, top: top,
                                               near: near, far: far)
        lightViewProjectionMatrix = lightProj
        
        // Shadow pass
        let shadowPassDesc = MTLRenderPassDescriptor()
        shadowPassDesc.depthAttachment.texture = shadowTexture
        shadowPassDesc.depthAttachment.loadAction = .clear
        shadowPassDesc.depthAttachment.storeAction = .store
        shadowPassDesc.depthAttachment.clearDepth = 1.0
        
        let shadowEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: shadowPassDesc)!
        var shadowTranslationMatrix = translationMatrix
        var shadowRotationMatrix = rotationMatrix
        var shadowScaleMatrix = scaleMatrix
        shadowEncoder.setDepthBias(4.0, slopeScale: 4.0, clamp: 0.0)
        shadowEncoder.setCullMode(.back)
        // shadowEncoder.setCullMode(.front)
        shadowEncoder.setFrontFacing(.counterClockwise)
        shadowEncoder.setRenderPipelineState(shadowPipelineState!)
        shadowEncoder.setDepthStencilState(depthStencilState)
        shadowEncoder.setViewport(MTLViewport(originX: 0, originY: 0, width: 1024, height: 1024, znear: 0, zfar: 1))
        shadowEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        shadowEncoder.setVertexBuffer(positionsBuffer, offset: 0, index: 1)
        shadowEncoder.setVertexBytes(&shadowTranslationMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 2)
        shadowEncoder.setVertexBytes(&shadowRotationMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 3)
        shadowEncoder.setVertexBytes(&shadowScaleMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 4)
        shadowEncoder.setVertexBytes(&lightViewProjectionMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 5)
        shadowEncoder.drawIndexedPrimitives(type: .triangle, indexCount: indices.count, indexType: isObj ? objSubmesh!.indexType : .uint16, indexBuffer: indexBuffer, indexBufferOffset: 0)
        shadowEncoder.endEncoding()
        
        // Main render pass
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor)!
        encoder.setRenderPipelineState(pipelineState)
        encoder.setDepthStencilState(depthStencilState)
        
        // Set buffers
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(positionsBuffer, offset: 0, index: 1)
        encoder.setVertexBuffer(normalsBuffer, offset: 0, index: 2)
        encoder.setVertexBuffer(uvBuffer, offset: 0, index: 3)
        
        // Uncomment to optimize by not drawing hidden faces when not rotating
        // encoder.setCullMode(.front)
        
        // Send the transformations matrix directly to the GPU without creating a persistent buffer (efficient for small, frequently updated data) with setVertexBytes
        encoder.setVertexBytes(&translationMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 4)
        encoder.setVertexBytes(&rotationMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 5)
        encoder.setVertexBytes(&scaleMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 6)
        encoder.setVertexBytes(&finalMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 7)
        encoder.setVertexBytes(&lightViewProjectionMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 8)

        if let texture = fillTexture {
            encoder.setFragmentTexture(texture, index: 0)
        }
        
        var useAllTextures = true

        if let displacementTexture = displacementTexture {
            encoder.setFragmentTexture(displacementTexture, index: 1)
        } else {
            useAllTextures = false
        }
        
        if let normalTexture = normalTexture {
            encoder.setFragmentTexture(normalTexture, index: 2)
        } else {
            useAllTextures = false
        }
        
        if let roughnessTexture = roughnessTexture {
            encoder.setFragmentTexture(roughnessTexture, index: 3)
        } else {
            useAllTextures = false
        }
        
        if let HDRPmaskMapTexture = HDRPmaskMapTexture {
            encoder.setFragmentTexture(HDRPmaskMapTexture, index: 4)
        } else {
            useAllTextures = false
        }
        
        encoder.setFragmentBytes(&useAllTextures, length: MemoryLayout<Bool>.size, index: 5)
        
        encoder.setFragmentBytes(&light, length: MemoryLayout<Light>.stride, index: 6)
        encoder.setFragmentBytes(&material, length: MemoryLayout<Material>.stride, index: 7)
        
        if let shadowTexture = shadowTexture {
            encoder.setFragmentTexture(shadowTexture, index: 8)
        }
        
        if let skyboxTexture = skyboxTexture {
            encoder.setFragmentTexture(skyboxTexture, index: 9)
        }
        
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indices.count,
            indexType: isObj ? objSubmesh!.indexType : .uint16,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )

        if let planeVertexBuffer = planeVertexBuffer, let planePositionsBuffer = planePositionsBuffer, let planeNormalsBuffer = planeNormalsBuffer, let planeUVBuffer = planeUVBuffer, let planeIndexBuffer = planeIndexBuffer {
            encoder.setVertexBuffer(planeVertexBuffer, offset: 0, index: 0)
            encoder.setVertexBuffer(planePositionsBuffer, offset: 0, index: 1)
            encoder.setVertexBuffer(planeNormalsBuffer, offset: 0, index: 2)
            encoder.setVertexBuffer(planeUVBuffer, offset: 0, index: 3)
            // Use identity matrices
            var planeTranslationMatrix = matrix_identity_float4x4
            var planeRotationMatrix = matrix_identity_float4x4
            var planeScaleMatrix = matrix_identity_float4x4
            var planeFinalMatrix = projectionMatrix * planeTranslationMatrix * planeRotationMatrix * planeScaleMatrix
            encoder.setVertexBytes(&planeTranslationMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 4)
            encoder.setVertexBytes(&planeRotationMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 5)
            encoder.setVertexBytes(&planeScaleMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 6)
            encoder.setVertexBytes(&planeFinalMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 7)
            encoder.setVertexBytes(&lightViewProjectionMatrix, length: MemoryLayout<matrix_float4x4>.stride, index: 8)
            if let planeFillTexture = planeFillTexture {
                encoder.setFragmentTexture(planeFillTexture, index: 0)
            }
            encoder.setFragmentBytes(&useAllTextures, length: MemoryLayout<Bool>.size, index: 5)
            encoder.setFragmentBytes(&light, length: MemoryLayout<Light>.stride, index: 6)
            encoder.setFragmentBytes(&planeMaterial, length: MemoryLayout<Material>.stride, index: 7)
            if let shadowTexture = shadowTexture {
                encoder.setFragmentTexture(shadowTexture, index: 8)
            }
            if let skyboxTexture = skyboxTexture {
                encoder.setFragmentTexture(skyboxTexture, index: 9)
            }
            
            encoder.drawIndexedPrimitives(type: .triangle, indexCount: planeIndices.count, indexType: .uint16, indexBuffer: planeIndexBuffer, indexBufferOffset: 0)
        }

        // Skybox pass
        if let skyboxPipelineState = skyboxPipelineState, let skyboxTexture = skyboxTexture {
            encoder.setRenderPipelineState(skyboxPipelineState)
            let skyboxDepthDescriptor = MTLDepthStencilDescriptor()
            skyboxDepthDescriptor.isDepthWriteEnabled = false
            skyboxDepthDescriptor.depthCompareFunction = .lessEqual
            let skyboxDepthState = device.makeDepthStencilState(descriptor: skyboxDepthDescriptor)
            encoder.setDepthStencilState(skyboxDepthState)
            encoder.setFragmentTexture(skyboxTexture, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36)
            encoder.setDepthStencilState(depthStencilState)
        }

        // 4. End & commit
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
