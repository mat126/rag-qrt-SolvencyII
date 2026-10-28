-- Tabella del feedback per la versione web (web/index.html).
-- Da eseguire una volta in Supabase: SQL Editor -> New query -> incolla -> Run.
--
-- Sicurezza: la chiave pubblica (anon/publishable) e' scritta nella pagina web, quindi
-- chiunque puo' leggerla. Le policy qui sotto le permettono SOLO di inserire righe:
-- niente lettura, modifica o cancellazione. Il feedback si legge con la chiave segreta
-- (service_role/secret), da usare solo in Python e mai nel browser.

create table if not exists public.feedback (
  id            uuid primary key default gen_random_uuid(),
  created_at    timestamptz not null default now(),
  question      text not null check (char_length(question) between 1 and 2000),
  annex         text check (annex is null or char_length(annex) <= 10),
  k             int  check (k is null or k between 1 and 50),
  mode          text check (mode is null or char_length(mode) <= 20),
  lang          text check (lang is null or char_length(lang) <= 5),
  rating        text not null check (rating in ('up', 'down')),
  expected      text check (expected is null or char_length(expected) <= 200),
  note          text check (note is null or char_length(note) <= 1000),
  hits          jsonb check (hits is null or (jsonb_typeof(hits) = 'array' and pg_column_size(hits) < 20000)),
  index_version text check (index_version is null or char_length(index_version) <= 40),
  client_time   text check (client_time is null or char_length(client_time) <= 40)
);

alter table public.feedback enable row level security;

-- Solo inserimento per i visitatori anonimi
drop policy if exists "anon can insert feedback" on public.feedback;
create policy "anon can insert feedback"
  on public.feedback for insert
  to anon
  with check (true);

-- Permessi espliciti: inserimento si', lettura/modifica/cancellazione no
grant usage on schema public to anon;
grant insert on public.feedback to anon;
revoke select, update, delete on public.feedback from anon;
