-- Update handle_new_user function to extract child details from auth metadata
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
  INSERT INTO public.users (
    id,
    email,
    full_name,
    child_id,
    age,
    grade,
    spelling_level
  )
  VALUES (
    new.id,
    new.email,
    new.raw_user_meta_data->>'full_name',
    new.raw_user_meta_data->>'child_id',
    (new.raw_user_meta_data->>'age')::integer,
    new.raw_user_meta_data->>'grade',
    new.raw_user_meta_data->>'spelling_level'
  )
  ON CONFLICT (id) DO NOTHING;
  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
