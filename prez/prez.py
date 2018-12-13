model.add(Convolution2D(1, # filter rétegek száma
                        3, 3, # 3x3 kernel méret
                        strides=(1,1) # lépés
                        input_shape=image))

