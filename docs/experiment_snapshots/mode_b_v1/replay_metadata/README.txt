Mode-B practical-CSI deterministic replay

Load: case = torch.load('cases/<name>.pt', map_location='cpu', weights_only=True)
All tensor leaves are CPU tensors, with original dtypes. Axes/shapes/bytes are in
case['metadata']['tensor_shapes_and_axes']. No precision reduction is performed.
Inputs y/h_hat/err_var/h_true have explicit singleton receiver axes. qam_symbols
keeps the singleton stream-per-UE axis. A1 inputs are ONLY inputs['a1_inputs']:
y and h are whitened by sqrt(N0 + sum_UE err_var), ruu is identity. Original Ruu
is N0*I. Truth/labels are diagnostic exports and never enter A1 inference.

Each methods[name] includes evaluator_llr and decoder_input_llr, per-UE error
counts, block-error booleans, CRC outcomes and decoded TB input bits. LLRs are
log P1/P0; hard bits are llr>0. decoder_input_llr has axes [B,UE,1,G]. The codec
squeezes only axis 2 before TBDecoder. No external clipping/scaling is applied.
Use the original NRTransportBlockCodec with manifest codec.summary fields and
Ndata=G/Qm, UE=16, on the recorded Sionna version; codec.decode(decoder_input_llr)
must reproduce decoded_payload_bits and crc_ok. Every exported file is reloaded
and decoded in the exporting environment to verify this property.

manifest.json records actual runtime codec/CRC/LDPC attributes, versions,
config, model/prior hashes and SHA256 for EVERY generated .pt. Some codec fields
are private runtime attributes and may be absent on other Sionna versions.
No claim about CRC membership in info_bits is inferred from names or constants.

Each physical input is regenerated twice and compared bitwise. All integer
error counts (including per-UE TB errors) are compared to the source JSON;
disagreements fail the export and leave a manifest with status=failed. Historical
raw tensors were not stored in that JSON: count agreement does not establish
bitwise identity with the historical run. Current/original environments are
recorded separately. Accept only a manifest with status=complete.

Optional A1 populations use original detector_for/prepare/sample_population/
readout APIs. Repeated K256 and exact-prefix K64 readouts must match exported
LLRs bitwise. Candidate symbols use the original constellation lookup; original
tokens/energies/log q/RB probabilities are also retained. Large unrequested
conditional per-candidate moments are omitted explicitly, not downcast.

storage in manifest.json projects 8 case files plus 2 diagnostic files from
actual saved sizes. Results are local-only and ignored by Git. No model training,
upload, weight copying, checkpoint selection or source-result modification occurs.
