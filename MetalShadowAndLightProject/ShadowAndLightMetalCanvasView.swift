//
//  ShadowAndLightMetalCanvasView.swift
//  MetalShadowAndLightProject
//
//  Created by Michaël ATTAL on 16/04/2025.
//

import MetalKit
import SwiftUI

// Projet Ombre et lumière - OpenGL Avancée - Version Metal

struct ShadowAndLightMetalCanvasView: NSViewRepresentable {
    enum ModelType {
        case staticDragon
        case loadedOBJ(URL)
    }
    
    let modelType: ModelType

    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero)
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.clearColor = MTLClearColorMake(0, 0, 0, 1)
        
        let renderer: ShadowAndLightMetalMetalRenderer
        switch modelType {
        case .staticDragon:
            renderer = ShadowAndLightMetalMetalRenderer(mtkView: mtkView)
        case .loadedOBJ(let url):
            renderer = ShadowAndLightMetalMetalRenderer(mtkView: mtkView, objURL: url)
        }
        mtkView.delegate = renderer
        
        context.coordinator.renderer = renderer
        
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var renderer: ShadowAndLightMetalMetalRenderer?
    }
}
