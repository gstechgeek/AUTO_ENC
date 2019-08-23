import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, isize, nz, ndf, nc, n_extra_layers=0):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize must be a multiple of 16"
        
        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 2, 2, 0, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf), nn.ReLU())
        csize, cnf = isize / 2, ndf

        # For Extra Layers
        for t in range(n_extra_layers):
            main.add_module('extra-conv-{0}-{1}'.format(t, ndf),
                        nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            main.add_module('extra-{0}-{1}-batchnorm'.format(t, ndf),
                        nn.BatchNorm2d(ndf))
            main.add_module('extra-relu-{0}-{1}'.format(t, ndf), nn.ReLU())
            # main.add_module('extra-maxpool-{0}-{1}'.format(t, ndf), nn.MaxPool2d(2, 2))

        while csize > 8:
            in_feat = cnf
            out_feat = cnf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 2, 2, 0, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.ReLU())
          # main.add_module('pyramid-{0}-maxpool'.format(out_feat),
          #                  nn.MaxPool2d(2, 2))
            cnf = cnf * 2 
            csize = csize / 2


        # main.add_module('final-{0}-{1}-conv'.format(cnf, 1),
        #                nn.Conv2d(cnf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output
        
class Decoder(nn.Module):
    def __init__(self, isize, nz, ngf, nc, n_extra_layers=0):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 8
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        """
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))
        """

        csize, _ = 8, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 2, 2, 0, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

class MSCDAE(nn.Module):
    def __init__(self, opt):
        super(MSCDAE, self).__init__()
        self.encoder = Encoder(opt.isize, opt.nz, opt.ndf, opt.nc, opt.n_extra_layers)
        self.decoder = Decoder(opt.isize, opt.nz, opt.ngf, opt.nc, opt.n_extra_layers)

    def forward(self, input):
        latent_vec  = self.encoder(input)
        output_tnsr = self.decoder(latent_vec)
        return output_tnsr
