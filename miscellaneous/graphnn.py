class NCF(nn.Module):
    def __init__(self, nb_users, nb_products, nb_latents, nb_mlbs, nb_hiddens, **kwargs):
        super(NCF, self).__init__(**kwargs)
        self.U = nn.Embedding(nb_users, nb_latents)
        self.Q = nn.Embedding(nb_users, nb_latents)  # latent user
        self.P = nn.Embedding(nb_products, nb_latents)
        self.V = nn.Embedding(nb_products, nb_latents)  # latent prod
        self.linears = nn.ModuleList(
            sum([(nn.ReLU(), nn.Linear(*nb_hiddens)) for i in range(nb_mlbs)], ()))
        self.pred = nn.ModuleList(
            [nn.ReLU(), nn.Linear(nb_hiddens[1], 1), nn.Sigmoid()])

    def forward(self, user_id, product_id):
        u_emb = self.U(user_id)
        p_emb = self.P(product_id)
        gmf = u_emb * p_emb
        u_latent = self.Q(user_id)
        p_latent = self.V(product_id)
        mlp = self.linears(torch.cat([p_latent, u_latent], dim=1))
        con_res = torch.cat([gmf, mlp], dim=1)
        return self.pred(con_res)