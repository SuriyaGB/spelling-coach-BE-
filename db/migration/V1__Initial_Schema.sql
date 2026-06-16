-- 1. EXTENDED USERS TABLE (Linked to Supabase Auth)
create table if not exists public.users (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
  theme_preference text default 'default',
  audio_enabled boolean default true
);

-- Enable RLS for users
alter table public.users enable row level security;
drop policy if exists "Users can view own profile" on public.users;
create policy "Users can view own profile" on public.users for select using (auth.uid() = id);
drop policy if exists "Users can update own profile" on public.users;
create policy "Users can update own profile" on public.users for update using (auth.uid() = id);

-- Trigger to automatically create a profile in public.users when a user signs up
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.users (id, email, full_name)
  values (new.id, new.email, new.raw_user_meta_data->>'full_name')
  on conflict (id) do nothing;
  return new;
end;
$$ language plpgsql security definer;

create or replace trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();


-- 2. CHILD PROFILES
create table if not exists public.child_profiles (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null references public.users(id) on delete cascade,
  child_id text not null,
  age integer,
  grade text,
  spelling_level text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.child_profiles enable row level security;
drop policy if exists "Users can manage child profiles" on public.child_profiles;
create policy "Users can manage child profiles" on public.child_profiles
  for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);
create index if not exists idx_child_profiles_user_child on public.child_profiles(user_id, child_id);


-- 3. CUSTOM WORD LISTS
create table if not exists public.custom_word_lists (
  id uuid default gen_random_uuid() primary key,
  owner_user_id uuid not null,
  name text not null,
  words jsonb not null default '[]'::jsonb,
  word_count integer default 0,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.custom_word_lists enable row level security;
drop policy if exists "Users can manage custom lists" on public.custom_word_lists;
create policy "Users can manage custom lists" on public.custom_word_lists
  for all to authenticated using (auth.uid() = owner_user_id) with check (auth.uid() = owner_user_id);
create index if not exists idx_custom_lists_owner on public.custom_word_lists(owner_user_id);


-- 4. USER SUBSCRIPTIONS
create table if not exists public.user_subscriptions (
  user_id uuid primary key references auth.users(id) on delete cascade,
  stripe_customer_id text,
  stripe_subscription_id text,
  status text,
  current_period_end timestamp with time zone,
  cancel_at_period_end boolean default false,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.user_subscriptions enable row level security;
drop policy if exists "Users can manage subscription status" on public.user_subscriptions;
create policy "Users can manage subscription status" on public.user_subscriptions
  for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);


-- 5. PRACTICE SESSIONS
create table if not exists public.practice_sessions (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null references public.users(id) on delete cascade,
  child_profile_id uuid references public.child_profiles(id) on delete cascade,
  mode text not null,
  session_started_at timestamp with time zone default timezone('utc'::text, now()) not null,
  session_ended_at timestamp with time zone,
  total_words_attempted integer default 0,
  total_correct integer default 0,
  accuracy_percentage decimal,
  duration_seconds integer,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.practice_sessions enable row level security;
drop policy if exists "Users can manage practice sessions" on public.practice_sessions;
create policy "Users can manage practice sessions" on public.practice_sessions
  for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);
create index if not exists idx_sessions_user_start on public.practice_sessions(user_id, session_started_at);
create index if not exists idx_sessions_child on public.practice_sessions(child_profile_id);


-- 6. WORD ATTEMPTS
create table if not exists public.word_attempts (
  id uuid default gen_random_uuid() primary key,
  session_id uuid references public.practice_sessions(id) on delete cascade,
  user_id uuid not null references public.users(id) on delete cascade,
  target_word text not null,
  child_attempt text not null,
  is_correct boolean not null,
  near_miss boolean default false,
  attempt_number integer not null default 1,
  error_types jsonb,
  supports_used jsonb,
  teaching_strategy text,
  confidence_level integer,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.word_attempts enable row level security;
drop policy if exists "Users can manage word attempts" on public.word_attempts;
create policy "Users can manage word attempts" on public.word_attempts
  for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);
create index if not exists idx_attempts_session on public.word_attempts(session_id);
create index if not exists idx_attempts_user_word on public.word_attempts(user_id, target_word);
create index if not exists idx_attempts_created_at on public.word_attempts(created_at);


-- 7. WORD METADATA (CACHE)
create table if not exists public.word_metadata (
  id uuid default gen_random_uuid() primary key,
  word_text text unique not null,
  definition text,
  example_sentence text,
  part_of_speech text,
  pronunciation text,
  origin text,
  difficulty_level text,
  spelling_patterns jsonb,
  prefixes jsonb,
  suffixes jsonb,
  greek_latin_roots jsonb,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.word_metadata enable row level security;
drop policy if exists "Anyone authenticated can read word metadata" on public.word_metadata;
create policy "Anyone authenticated can read word metadata" on public.word_metadata for select to authenticated using (true);
create index if not exists idx_word_metadata_text on public.word_metadata(word_text);


-- 8. USER STATISTICS
create table if not exists public.user_statistics (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null references public.users(id) on delete cascade,
  child_profile_id uuid references public.child_profiles(id) on delete cascade,
  total_sessions integer default 0,
  total_attempts integer default 0,
  total_correct integer default 0,
  current_streak integer default 0,
  longest_streak integer default 0,
  average_accuracy decimal default 0,
  words_mastered integer default 0,
  last_practice_date timestamp with time zone,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.user_statistics enable row level security;
drop policy if exists "Users can manage own statistics" on public.user_statistics;
create policy "Users can manage own statistics" on public.user_statistics
  for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);
create index if not exists idx_stats_user on public.user_statistics(user_id);


-- 9. FOREIGN ORIGIN WORD LISTS (CACHE)
create table if not exists public.foreign_origin_word_lists (
  id uuid default gen_random_uuid() primary key,
  origin_language text not null,
  word text not null,
  definition text,
  category text,
  difficulty text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

alter table public.foreign_origin_word_lists enable row level security;
drop policy if exists "Anyone authenticated can read foreign origin lists" on public.foreign_origin_word_lists;
create policy "Anyone authenticated can read foreign origin lists" on public.foreign_origin_word_lists for select to authenticated using (true);
create index if not exists idx_foreign_origins_language on public.foreign_origin_word_lists(origin_language);
