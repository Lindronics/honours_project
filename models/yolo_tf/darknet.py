import tensorflow as tf
import layers

def darknet(x):

    with tf.variable_scope("darknet"):

        inputs, _ = layers.conv_layer(name="darknet_conv_0", inputs=inputs, filters=32, kernel_size=3)
        inputs, shortcut = layers.conv_layer(name="darknet_conv_1", inputs=inputs, filters=64, kernel_size=3, downsample=True)
        

        for i in range(1):
            inputs, shortcut = layers.residual_block(f"residual_{i}", inputs, num_filters=32)
        
        inputs, shortcut = layers.conv_layer(name="darknet_conv_2", inputs=inputs, filters=128, kernel_size=3, downsample=True)
        
        
        for i in range(2):
            inputs, shortcut = layers.residual_block(f"residual_{i + 1}", inputs, num_filters=64)
        
        inputs, shortcut = layers.conv_layer(name="darknet_conv_3", inputs=inputs, filters=256, kernel_size=3, downsample=True)
        
        
        for i in range(8):
            inputs, darknet_route_1 = layers.residual_block(f"residual_{i + 3}", inputs, num_filters=128)
        
        inputs, shortcut = layers.conv_layer(name="darknet_conv_4", inputs=inputs, filters=512, kernel_size=3, downsample=True)


        for i in range(8):
            inputs, darknet_route_2 = layers.residual_block(f"residual_{i + 11}", inputs, num_filters=256)

        inputs, shortcut = layers.conv_layer(name="darknet_conv_5", inputs=inputs, filters=1024, kernel_size=3, downsample=True)


        for i in range(4):
            inputs, shortcut = layers.residual_block(f"residual_{i + 19}", inputs, num_filters=512)
            

        return darknet_route_1, darknet_route_2, inputs
