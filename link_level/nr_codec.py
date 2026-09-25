"""Native NR TB codec with explicit [batch, UE, codeword] ordering."""

from link_level.sionna_ebno import effective_payload_rate, require_sionna2


def symbols_to_grid_input(symbols):
    if symbols.ndim != 3:
        raise ValueError("Expected symbols [B,K,Ndata].")
    return symbols.unsqueeze(2)


def grid_llrs_to_codewords(llr, num_ues, coded_bits):
    if llr.ndim != 4 or tuple(llr.shape[1:]) != (num_ues, 1, coded_bits):
        raise ValueError(f"Expected LLR [B,{num_ues},1,{coded_bits}], got {tuple(llr.shape)}.")
    return llr.squeeze(2).contiguous()


class NRTransportBlockCodec:
    def __init__(self, num_data_symbols, num_ues, table=1, index=10,
                 num_bp_iter=20, precision="single", device="cpu"):
        require_sionna2()
        from sionna.phy.mapping import Mapper
        from sionna.phy.nr import TBEncoder, TBDecoder
        from sionna.phy.nr.utils import calculate_tb_size
        from link_level.nr_mcs import get_pusch_mcs

        if num_data_symbols <= 0 or not 1 <= num_ues <= 65535 or num_bp_iter <= 0:
            raise ValueError("Invalid grid, UE count, or BP iteration count.")
        self.mcs = get_pusch_mcs(table, index, device=device)
        self.num_ues, self.num_data_symbols = int(num_ues), int(num_data_symbols)
        self.qm = self.mcs.bits_per_symbol
        self.coded_bits = self.num_data_symbols * self.qm
        tb_size = int(calculate_tb_size(
            modulation_order=self.qm, target_coderate=self.mcs.target_coderate,
            num_coded_bits=self.coded_bits, num_layers=1, device=device)[0])
        self.encoder = TBEncoder(
            target_tb_size=tb_size, num_coded_bits=self.coded_bits,
            target_coderate=self.mcs.target_coderate, num_bits_per_symbol=self.qm,
            num_layers=1, n_rnti=list(range(1, self.num_ues + 1)), n_id=[1] * self.num_ues,
            channel_type="PUSCH", use_scrambler=True, precision=precision, device=device)
        self.info_bits = int(self.encoder.k)
        if int(self.encoder.n) != self.coded_bits or self.encoder.k_padding != 0:
            raise RuntimeError("Unexpected TB padding/coded length; resolve codec accounting first.")
        self.rate = effective_payload_rate(self.info_bits, self.coded_bits)
        self.decoder = TBDecoder(self.encoder, num_bp_iter=num_bp_iter,
                                 precision=precision, device=device)
        self.mapper = Mapper("qam", self.qm, precision=precision, device=device)
        self.num_bp_iter = num_bp_iter

    def encode(self, info):
        if info.ndim != 3 or tuple(info.shape[1:]) != (self.num_ues, self.info_bits):
            raise ValueError("Expected information bits [B,K,k].")
        coded = self.encoder(info)
        if tuple(coded.shape) != (info.shape[0], self.num_ues, self.coded_bits):
            raise RuntimeError("Unexpected encoder output shape.")
        return coded, symbols_to_grid_input(self.mapper(coded))

    def decode(self, llr):
        codewords = grid_llrs_to_codewords(llr, self.num_ues, self.coded_bits)
        bits, crc = self.decoder(codewords)
        if tuple(bits.shape) != (llr.shape[0], self.num_ues, self.info_bits):
            raise RuntimeError("Unexpected decoder output shape.")
        return bits, crc.reshape(llr.shape[0], self.num_ues)

    def metadata(self):
        return dict(mcs_table=self.mcs.table, mcs_index=self.mcs.index,
                    bits_per_symbol=self.qm, target_coderate=self.mcs.target_coderate,
                    payload_coderate=self.rate, info_bits_per_ue=self.info_bits,
                    coded_bits_per_ue=self.coded_bits, tb_size=int(self.encoder.tb_size),
                    padding_bits=int(self.encoder.k_padding), num_bp_iter=self.num_bp_iter,
                    llr_convention="log(P(bit=1)/P(bit=0))", scrambling=True)
