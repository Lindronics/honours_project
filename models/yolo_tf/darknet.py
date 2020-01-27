import tensorflow as tf
import layers

def darknet(inputs):

    with tf.name_scope("darknet"):

        inputs = layers.conv_layer(name="darknet/conv_0", inputs=inputs, filters=32, kernel_size=3)
        inputs = layers.conv_layer(name="darknet/conv_1", inputs=inputs, filters=64, kernel_size=3, downsample=True)
        

        for i in range(1):
            inputs = layers.residual_block(f"darknet/residual_group_0/residual_{i}", inputs, num_filters=32)
        
        inputs = layers.conv_layer(name="darknet/residual_group_0/conv_2", inputs=inputs, filters=128, kernel_size=3, downsample=True)
        
        
        for i in range(2):
            inputs = layers.residual_block(f"darknet/residual_group_1/residual_{i}", inputs, num_filters=64)
        
        inputs = layers.conv_layer(name="darknet/residual_group_1/conv_3", inputs=inputs, filters=256, kernel_size=3, downsample=True)
        
        
        for i in range(8):
            inputs = layers.residual_block(f"darknet/residual_group_2/residual_{i}", inputs, num_filters=128)
        
        darknet_route_1 = inputs
        inputs = layers.conv_layer(name="darknet/residual_group_3/conv_4", inputs=inputs, filters=512, kernel_size=3, downsample=True)


        for i in range(8):
            inputs = layers.residual_block(f"darknet/residual_group_3/residual_{i}", inputs, num_filters=256)

        darknet_route_2 = inputs
        inputs = layers.conv_layer(name="darknet/residual_group_4/conv_5", inputs=inputs, filters=1024, kernel_size=3, downsample=True)


        for i in range(4):
            inputs = layers.residual_block(f"darknet/residual_group_4/residual_{i}", inputs, num_filters=512)
            

        return darknet_route_1, darknet_route_2, inputs
