from omegaconf import OmegaConf


def register_resolvers():
    OmegaConf.register_new_resolver("not", lambda boolean: not boolean, replace=True)

    OmegaConf.register_new_resolver(
        "equal", lambda arg1, arg2: arg1 == arg2, replace=True
    )

    def conditional_resolver(condition, arg1, arg2):
        return arg1 if condition else arg2

    OmegaConf.register_new_resolver("ifelse", conditional_resolver, replace=True)
