import keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


def hu_classification_model(
        num_classes,
        num_anchors):

    options1 = {
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }
    options3 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs_P5 = keras.layers.Input(shape=(512, None, None))
        inputs_P4 = keras.layers.Input(shape=(256, None, None))
        inputs_P3 = keras.layers.Input(shape=(128, None, None))
        # inputs_P2 = keras.layers.Input(shape=(64, None, None))
    else:
        inputs_P5 = keras.layers.Input(shape=(15, 20, 2048))
        inputs_P4 = keras.layers.Input(shape=(30, 40, 1024))
        inputs_P3 = keras.layers.Input(shape=(60, 80, 512))
        # inputs_P2 = keras.layers.Input(shape=(None, None, 64))

    inputs = [inputs_P3, inputs_P4, inputs_P5]

    D5 = keras.layers.Conv2D(512, **options1)(inputs_P5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(1024, **options3)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(512, **options1)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(1024, **options3)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(512, **options1)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5_up = keras.layers.Conv2DTranspose(1024, kernel_size=2, strides=2, padding='valid')(D5)
    D4 = keras.layers.Add()([D5_up, inputs_P4])

    D4 = keras.layers.Conv2D(256, **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(512, **options3)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(256,  **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(512, **options3)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(128, **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4_up = keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='valid')(D4)
    D3 = keras.layers.Add()([D4_up, inputs_P3])

    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options3)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options3)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    outputs = keras.layers.Conv2D(filters=num_classes * num_anchors, **options1)(D3)

    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs)
    outputs = keras.layers.Reshape((-1, num_classes))(outputs)

    outputs = keras.layers.Activation('sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name='cls')  # , name=name)


def hu_mask_model(
        num_classes,
        num_anchors):

    options1 = {
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }
    options3 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs_P5 = keras.layers.Input(shape=(512, None, None))
        inputs_P4 = keras.layers.Input(shape=(256, None, None))
        inputs_P3 = keras.layers.Input(shape=(128, None, None))
        # inputs_P2 = keras.layers.Input(shape=(64, None, None))
    else:
        inputs_P5 = keras.layers.Input(shape=(15, 20, 2048))
        inputs_P4 = keras.layers.Input(shape=(30, 40, 1024))
        inputs_P3 = keras.layers.Input(shape=(60, 80, 512))
        # inputs_P2 = keras.layers.Input(shape=(None, None, 64))

    inputs = [inputs_P3, inputs_P4, inputs_P5]

    D5 = keras.layers.Conv2D(512, **options1)(inputs_P5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(1024, **options3)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(512, **options1)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(1024, **options3)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(512, **options1)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5_up = keras.layers.Conv2DTranspose(1024, kernel_size=2, strides=2, padding='valid')(D5)
    D4 = keras.layers.Add()([D5_up, inputs_P4])

    D4 = keras.layers.Conv2D(256, **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(512, **options3)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(256,  **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(512, **options3)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(128, **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4_up = keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='valid')(D4)
    D3 = keras.layers.Add()([D4_up, inputs_P3])

    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options3)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options3)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    outputs = keras.layers.Conv2D(filters=num_classes, **options1)(D3)

    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs)
    outputs = keras.layers.Reshape((-1, num_classes))(outputs)

    outputs = keras.layers.Activation('sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name='seg')  # , name=name)


def hu_regression_model(num_values, num_anchors):
    options1 = {
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }
    options3 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }

    inputs_P5 = keras.layers.Input(shape=(15, 20, 2048))
    inputs_P4 = keras.layers.Input(shape=(30, 40, 1024))
    inputs_P3 = keras.layers.Input(shape=(60, 80, 512))
    # inputs_P2 = keras.layers.Input(shape=(120, 160, 64))

    inputs = [inputs_P3, inputs_P4, inputs_P5]

    D5 = keras.layers.Conv2D(512, **options1)(inputs_P5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(1024, **options3)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(512, **options1)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(1024, **options3)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5 = keras.layers.Conv2D(512, **options1)(D5)
    D5 = keras.layers.LeakyReLU(alpha=0.1)(D5)
    D5_up = keras.layers.Conv2DTranspose(1024, kernel_size=2, strides=2, padding='valid')(D5)
    D4 = keras.layers.Add()([D5_up, inputs_P4])

    D4 = keras.layers.Conv2D(256, **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(512, **options3)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(256, **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(512, **options3)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4 = keras.layers.Conv2D(128, **options1)(D4)
    D4 = keras.layers.LeakyReLU(alpha=0.1)(D4)
    D4_up = keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='valid')(D4)
    D3 = keras.layers.Add()([D4_up, inputs_P3])

    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options3)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options3)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(128, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)
    D3 = keras.layers.Conv2D(256, **options1)(D3)
    D3 = keras.layers.LeakyReLU(alpha=0.1)(D3)

    outputs = keras.layers.Conv2D(filters=num_values * num_anchors, **options1)(D3)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs)
    outputs = keras.layers.Reshape((-1, num_values))(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name='hyp')  # , name=name)


def default_classification_model(
        num_classes,
        num_anchors,
        prior_probability = 0.01,
):

    options3 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs_P5 = keras.layers.Input(shape=(1024, None, None))
        inputs_P4 = keras.layers.Input(shape=(512, None, None))
        inputs_P3 = keras.layers.Input(shape=(256, None, None))
        # inputs_P2 = keras.layers.Input(shape=(64, None, None))
    else:
        inputs_P5 = keras.layers.Input(shape=(15, 20, 1024))
        inputs_P4 = keras.layers.Input(shape=(30, 40, 512))
        inputs_P3 = keras.layers.Input(shape=(60, 80, 256))
        # inputs_P2 = keras.layers.Input(shape=(None, None, 64))

    inputs = [inputs_P3, inputs_P4, inputs_P5]

    D5 = keras.layers.Conv2D(1024, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(inputs_P5)
    #D5 = keras.layers.Conv2D(512, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    #    bias_initializer='zeros', **options3)(D5)
    D5_up = keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='valid')(D5)
    D4 = keras.layers.Add()([D5_up, inputs_P4])

    D4 = keras.layers.Conv2D(512, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(D4)
    #D4 = keras.layers.Conv2D(256, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    #    bias_initializer='zeros', **options3)(D4)
    D4_up = keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='valid')(D4)
    D3 = keras.layers.Add()([D4_up, inputs_P3])

    D3 = keras.layers.Conv2D(256, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(D3)
    D3 = keras.layers.Conv2D(256, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(D3)

    outputs = keras.layers.Conv2D(filters=num_classes * num_anchors,kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability), **options3)(D3)

    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs)
    outputs = keras.layers.Reshape((-1, num_classes))(outputs)

    outputs = keras.layers.Activation('sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name='cls')  # , name=name)


def default_mask_model(
        num_classes,
        num_anchors,
        prior_probability=0.01,
):

    options3 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs_P5 = keras.layers.Input(shape=(1024, None, None))
        inputs_P4 = keras.layers.Input(shape=(512, None, None))
        inputs_P3 = keras.layers.Input(shape=(256, None, None))
        # inputs_P2 = keras.layers.Input(shape=(64, None, None))
    else:
        inputs_P5 = keras.layers.Input(shape=(15, 20, 1024))
        inputs_P4 = keras.layers.Input(shape=(30, 40, 512))
        inputs_P3 = keras.layers.Input(shape=(60, 80, 256))
        # inputs_P2 = keras.layers.Input(shape=(None, None, 64))

    inputs = [inputs_P3, inputs_P4, inputs_P5]

    D5 = keras.layers.Conv2D(1024, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(inputs_P5)
    #D5 = keras.layers.Conv2D(512, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    #    bias_initializer='zeros', **options3)(D5)
    D5_up = keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='valid')(D5)
    D4 = keras.layers.Add()([D5_up, inputs_P4])

    D4 = keras.layers.Conv2D(512, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(D4)
    #D4 = keras.layers.Conv2D(256, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    #    bias_initializer='zeros', **options3)(D4)
    D4_up = keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='valid')(D4)
    D3 = keras.layers.Add()([D4_up, inputs_P3])

    D3 = keras.layers.Conv2D(256, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(D3)
    D3 = keras.layers.Conv2D(256, activation='relu',kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros', **options3)(D3)

    outputs = keras.layers.Conv2D(filters=num_classes,kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability), **options3)(D3)

    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs)
    outputs = keras.layers.Reshape((-1, num_classes))(outputs)

    outputs = keras.layers.Activation('sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name='seg')  # , name=name)


def default_regression_model(num_values, num_anchors):
    options3 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs_P5 = keras.layers.Input(shape=(1024, None, None))
        inputs_P4 = keras.layers.Input(shape=(512, None, None))
        inputs_P3 = keras.layers.Input(shape=(256, None, None))
        # inputs_P2 = keras.layers.Input(shape=(64, None, None))
    else:
        inputs_P5 = keras.layers.Input(shape=(15, 20, 1024))
        inputs_P4 = keras.layers.Input(shape=(30, 40, 512))
        inputs_P3 = keras.layers.Input(shape=(60, 80, 256))
        # inputs_P2 = keras.layers.Input(shape=(None, None, 64))

    inputs = [inputs_P3, inputs_P4, inputs_P5]

    D5 = keras.layers.Conv2D(1024, activation='relu', **options3)(inputs_P5)
    #D5 = keras.layers.Conv2D(512, activation='relu', **options3)(D5)
    D5_up = keras.layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='valid')(D5)
    D4 = keras.layers.Add()([D5_up, inputs_P4])

    D4 = keras.layers.Conv2D(512, activation='relu', **options3)(D4)
    #D4 = keras.layers.Conv2D(256, activation='relu', **options3)(D4)
    D4_up = keras.layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='valid')(D4)
    D3 = keras.layers.Add()([D4_up, inputs_P3])

    D3 = keras.layers.Conv2D(256, activation='relu', **options3)(D3)
    D3 = keras.layers.Conv2D(256, activation='relu', **options3)(D3)

    outputs = keras.layers.Conv2D(filters=num_values * num_anchors, **options3)(D3)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs)
    outputs = keras.layers.Reshape((-1, num_values))(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name='hyp')  # , name=name)


def default_submodels(num_classes, num_anchors):
    return [
        (default_regression_model(16, num_anchors)),
        (default_classification_model(num_classes, num_anchors)),
        (default_mask_model(num_classes, num_anchors)),
    ]


def hu_submodels(num_classes, num_anchors):
    return [
        (hu_regression_model(16, num_anchors)),
        (hu_classification_model(num_classes, num_anchors)),
        (hu_mask_model(num_classes, num_anchors)),
    ]


def __build_pyramid(models, features):
    model_hyp = models[0]
    model_cls = models[1]
    model_mask = models[2]

    models_hyp = model_hyp(features)
    models_cls = model_cls(features)
    models_mask = model_mask(features)

    return [models_hyp, models_cls, models_mask]


def __build_anchors(anchor_parameters, features):
    anchors = layers.Anchors(
        size=anchor_parameters.sizes[0],
        stride=anchor_parameters.strides[0],
        ratios=anchor_parameters.ratios,
        scales=anchor_parameters.scales,
        name='anchors_{}'.format(0)
    )(features[0])
    return anchors


def retinanet(
        inputs,
        backbone_layers,
        num_classes,
        num_anchors=None,
        submodels=None,
        name='retinanet'
):
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    C3, C4, C5 = backbone_layers

    C3 = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='P3')(C3)
    C4 = keras.layers.Conv2D(512, kernel_size=1, strides=1, padding='same', name='P4')(C4)
    C5 = keras.layers.Conv2D(1024, kernel_size=1, strides=1, padding='same', name='P5')(C5)

    pyramids = __build_pyramid(submodels, [C3, C4, C5])

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    anchor_params         = None,
    **kwargs
):

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    # compute the anchors
    #features = [model.get_layer(p_name).output for p_name in ['P3_con', 'P4_con', 'P5_con', 'P6_con', 'P7_con']]
    features = [model.get_layer(p_name).output for p_name in ['res3d_relu']]
    anchors = __build_anchors(anchor_params, features)

    regression3D = model.outputs[0]
    classification = model.outputs[1]
    mask = model.outputs[2]
    other = model.outputs[3:]

    boxes3D = layers.RegressBoxes3D(name='boxes3D')([anchors, regression3D])

    return keras.models.Model(inputs=model.inputs, outputs=[boxes3D, classification, mask], name=name)
