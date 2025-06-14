//
//  ContentView.swift
//  MetalShadowAndLightProject
//
//  Created by Michaël ATTAL on 16/04/2025.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        HStack {
            ShadowAndLightMetalCanvasView(modelType: .staticDragon).aspectRatio(contentMode: .fit)
            ShadowAndLightMetalCanvasView(modelType: .loadedOBJ(Bundle.main.url(forResource: "monkey", withExtension: "obj")!)).aspectRatio(contentMode: .fit)
        }
    }
}

#Preview {
    ContentView()
}
