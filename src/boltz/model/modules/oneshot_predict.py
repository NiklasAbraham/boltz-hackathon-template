

class Boltz2OneShot(nn.Module):
    """
    Drop-in replacement for AtomDiffusion / AtomDiffusionDeterministic.
    Uses the provided ScoreModel to directly denoise coordinates at a given sigma.
    """

    def __init__(
        self,
        score_model_args: dict,
        compile_score: bool = False,
        sigma_key: str = "sigma",
        coords_key: str = "coords_noised",  # noised input coordinates
        **ignored_diffusion_args,
    ):
        super().__init__()

        self.score_model = ScoreModel(score_model_args)

        if compile_score:
            self.score_model = torch.compile(self.score_model, dynamic=False, fullgraph=False)

        # names in feats[] to read sigma + noised coords
        self.sigma_key = sigma_key
        self.coords_key = coords_key

    # -------------------------
    # INFERENCE MODE (Boltz2 calls structure_module.sample(...))
    # -------------------------
    @torch.no_grad()
    def sample(
        self,
        s_trunk,
        s_inputs,
        feats,
        num_sampling_steps=None,      # ignored – kept for API compatibility
        atom_mask=None,
        multiplicity=1,
        max_parallel_samples=None,    # ignored
        steering_args=None,           # overewritten with noise input and sigma
        diffusion_conditioning=None,
    ):
        """
        Called by Boltz2 in predict() mode.
        Returns a dict containing sample_atom_coords.
        """

        # ---- read inputs from feats ----
        noised_coords = steering_args[self.coords_key]           # (B, L, 3)
        sigma = steering_args[self.sigma_key]            # (B,) or scalar

        # ---- run score model ----
        atom_coords = self.score_model(
            noised_atom_coords=noised_coords,
            sigma=sigma,
            network_condition_kwargs=dict(
                s_trunk=s_trunk,
                s_inputs=s_inputs,
                feats=feats,
                diffusion_conditioning=diffusion_conditioning,
            ),
        )

        # expand to multiplicity for compatibility
        if multiplicity > 1:
            atom_coords = atom_coords.unsqueeze(1).repeat(1, multiplicity, 1, 1)
        else:
            atom_coords = atom_coords.unsqueeze(1)

        return dict(sample_atom_coords=atom_coords, diff_token_repr=None,)

    # # -------------------------
    # # TRAINING MODE (Boltz2 calls structure_module(...))
    # # -------------------------
    # def forward(
    #     self,
    #     s_trunk,
    #     s_inputs,
    #     feats,
    #     multiplicity=1,
    #     diffusion_conditioning=None,
    # ):
    #     """
    #     Training path: must return same keys as AtomDiffusion.forward().
    #     """

    #     noised_coords = feats[self.coords_key]   # (B*m, L, 3) already expanded by Boltz2
    #     sigma         = feats[self.sigma_key]    # (B*m) or scalar

    #     denoised = self.score_model(
    #         noised_atom_coords=noised_coords,
    #         sigma=sigma,
    #         network_condition_kwargs=dict(
    #             s_trunk=s_trunk,
    #             s_inputs=s_inputs,
    #             feats=feats,
    #             diffusion_conditioning=diffusion_conditioning,
    #         ),
    #     )

    #     # Return exactly what training expects
    #     return {
    #         "pred_atom_coords": denoised,        # replaces predicted x_t
    #         "diff_token_repr": None,
    #     }

    # # -------------------------
    # # Utility for Boltz2 loss
    # # -------------------------
    # def compute_loss(self, batch, out, multiplicity, **kwargs):
    #     """
    #     Dummy loss (score model is not diffusion-trained here).
    #     You can define a proper regression loss if desired.
    #     """
    #     if "coords" in batch:
    #         target = batch["coords"]  # (B*m, L, 3)
    #         pred   = out["pred_atom_coords"]
    #         loss = ((pred - target) ** 2).mean()
    #         return {"loss": loss, "loss_breakdown": {"mse_loss": loss}}
    #     else:
    #         # no loss if no ground-truth coordinates present
    #         z = torch.tensor(0.0, device=pred.device, requires_grad=True)
    #         return {"loss": z, "loss_breakdown": {}}
