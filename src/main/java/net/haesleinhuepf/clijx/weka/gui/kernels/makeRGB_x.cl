__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void makeRGB(
IMAGE_srcInput_TYPE  srcInput,
IMAGE_srcForeground_TYPE  srcForeground,
IMAGE_srcBackground_TYPE  srcBackground,
IMAGE_dst_TYPE   dst,

float input_min,
float input_max,

float input_r,
float input_g,
float input_b,

float foreground_r,
float foreground_g,
float foreground_b,

float background_r,
float background_g,
float background_b

)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const int2 pos = (int2){x,y};

  const float input = (READ_IMAGE(srcInput, sampler, pos).x - input_min) * 255.0 / (input_max - input_min);
  const float foreground = READ_IMAGE(srcForeground, sampler, pos).x;
  const float background = READ_IMAGE(srcBackground, sampler, pos).x;

  const float value_r = input * input_r + foreground * foreground_r + background * background_r;
  const float value_g = input * input_g + foreground * foreground_g + background * background_g;
  const float value_b = input * input_b + foreground * foreground_b + background * background_b;


  const int4 pos_r = (int4){x, y, 0, 0};
  const int4 pos_g = (int4){x, y, 1, 0};
  const int4 pos_b = (int4){x, y, 2, 0};
  WRITE_dst_IMAGE (dst, pos_r, CONVERT_dst_PIXEL_TYPE(value_r));
  WRITE_dst_IMAGE (dst, pos_g, CONVERT_dst_PIXEL_TYPE(value_g));
  WRITE_dst_IMAGE (dst, pos_b, CONVERT_dst_PIXEL_TYPE(value_b));
}