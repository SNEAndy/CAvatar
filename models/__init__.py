from .avatar_facerecon import AvatarFaceRecon


def create_model(opt):
    model = AvatarFaceRecon(opt)
    return model
