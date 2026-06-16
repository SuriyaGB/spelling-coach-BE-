-- 1. Drop unused tables (word_metadata & foreign_origin_word_lists)
DROP TABLE IF EXISTS public.word_metadata CASCADE;
DROP TABLE IF EXISTS public.foreign_origin_word_lists CASCADE;

-- 2. Drop child_profile_id columns from other tables
ALTER TABLE public.practice_sessions DROP COLUMN IF EXISTS child_profile_id CASCADE;
ALTER TABLE public.user_statistics DROP COLUMN IF EXISTS child_profile_id CASCADE;

-- 3. Drop child_profiles table
DROP TABLE IF EXISTS public.child_profiles CASCADE;

-- 4. Add child columns directly to users table to merge them
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS child_id text;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS age integer;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS grade text;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS spelling_level text;
