namespace qcu {

template <typename _Float>
class EOPreconditioner {
 public:
  virtual void apply(Complex<float>* output, Complex<float>* input, const Latt_Desc& desc,
                     int site_vec_len, int Nd = 4, void* stream = nullptr);
  virtual void reverse(Complex<float>* output, Complex<float>* input, const Latt_Desc& desc,
                       int site_vec_len, int Nd = 4, void* stream = nullptr);
};

template <typename _Float>
class GaugeEOPreconditioner : public EOPreconditioner<_Float> {
 public:
  void apply(Complex<float>* output, Complex<float>* input, const Latt_Desc& desc, int site_vec_len,
             int Nd = 4, void* stream = nullptr) override;
  void reverse(Complex<float>* output, Complex<float>* input, const Latt_Desc& desc,
               int site_vec_len, int Nd = 4, void* stream = nullptr) override;
};
}