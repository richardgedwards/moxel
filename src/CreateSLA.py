
if __name__=="__main__":
    fileName = 'sph'
    machineType = 'default'
    estimatedPrintTime = 2217.2
    volume = 4.77357
    resin = 'normal'
    weight = 5.44187
    price = 0.286414
    layerHeight = 0.05
    resolutionX = 1440
    resolutionY = 2560
    machineX = 68.04
    machineY = 120.96
    machineZ = 155
    projectType = 'LCD_mirror'
    normalExposureTime = 10
    bottomLayerExposureTime = 35
    normalDropSpeed = 300
    normalLayerLiftHeight = 5
    zSlowUpDistance = 0
    normalLayerLiftSpeed = 150

    bottomLayerCount = 4
    mirror = 1
    totalLayer = 1010
    bottomLayerLiftHeight = 5
    bottomLayerLiftSpeed = 150
    bottomLightOffTime = 0
    lightOffTime = 0

    with open('./img/gcode.run', 'w') as f:
        f.write('G21;\n')
        f.write('G90;\n')
        f.write('M106 S0;\n')
        f.write('G28 Z0;\n')
        f.write('\n')

        rise_pos = 0
        for k in range(bottomLayerCount):
            f.write('M6054 "{:}{:04d}.png";\n'.format(fileName,k))
            f.write('G0 Z{:.2f} F{:d};\n'.format(rise_pos+bottomLayerLiftHeight, bottomLayerLiftSpeed))
            f.write('G0 Z{:.2f} F{:d};\n'.format(rise_pos+layer_height, normalDropSpeed))
            f.write('G4 P{:d};\n'.format(bottomLightOffTime))
            f.write('M106 S255;\n')
            f.write('G4 P{:d};\n'.format(bottomLayerExposureTime*1000))
            f.write('M106 S0;\n')
            f.write('\n')
            rise_pos += layer_height

        for k in range(bottomLayerCount, totalLayer):
            f.write('M6054 "{:}{:04d}.png";\n'.format(fileName,k))
            f.write('G0 Z{:.2f} F{:d};\n'.format(rise_pos+normalLayerLiftHeight, normalLayerLiftSpeed))
            f.write('G0 Z{:.2f} F{:d};\n'.format(rise_pos+layer_height, normalDropSpeed))
            f.write('G4 P{:d};\n'.format(lightOffTime))
            f.write('M106 S255;\n')
            f.write('G4 P{:d};\n'.format(normalExposureTime*1000))
            f.write('M106 S0;\n')
            f.write('\n')
            rise_pos += layer_height

        f.write('M106 S0;\n')
        f.write('G1 Z{:d} F25;\n'.format(machine_height))
        f.write('M18;\n')

    