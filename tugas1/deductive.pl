% Fakta
coffee(morning, espresso).
coffee(afternoon, latte).
coffee(evening, cappuccino).

% Aturan deduktif
recommend(Time, Coffee) :- coffee(Time, Coffee).
